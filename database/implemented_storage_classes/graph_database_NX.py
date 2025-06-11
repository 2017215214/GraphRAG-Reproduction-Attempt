from __future__ import annotations
import os
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple, Union, TYPE_CHECKING
import networkx as nx

from graspologic.partition import hierarchical_leiden

from database.BaseClasses import BaseGraphStorage
from database.data_format_specification.community_format import SingleCommunityFormat_NoReport
from operators.utils import logger
from Use_LLM.prompt import GRAPH_FIELD_SEP


# SingleCommunitySchema = Dict[str, Union[int, str, List[List[str]], List[str], float]]

@dataclass
class GraphDatabaseNetworkX(BaseGraphStorage):
    """
    Graph storage implementation based on NetworkX.
    """

    def __post_init__(self) -> None:
        # prepare file path
        self._graph_file: str = os.path.join(
            self.global_config["working_dir"], f'graph_{self.name_space}.graphml'
        )
        # load or init
        loaded = self.load_nx_graph(self._graph_file)
        if loaded is not None:
            logger.info(
                f"Loaded graph from {self._graph_file} with {loaded.number_of_nodes()} nodes, {loaded.number_of_edges()} edges"
            )
        self._graph: nx.Graph = loaded or nx.Graph()
        # algorithm registry
        self._clustering_algorithms: Dict[str, Any] = {
            'leiden': self._leiden_clustering,
        }
        # self._node_embed_algorithms: Dict[str, Any] = {
        #     'node2vec': self._node2vec_embed,
        # }

    # ---------------- I/O ----------------
    @staticmethod
    def load_nx_graph(path: str) -> Optional[nx.Graph]:
        """Load a GraphML file if exists, else return None."""
        if os.path.exists(path):
            return nx.read_graphml(path)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, path: str) -> None:
        """Write graph to GraphML file."""
        import os
        
        # 确保父目录存在
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created directory: {parent_dir}")
        
        logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges to {path}")
        nx.write_graphml(graph, path)

    async def index_start_callback(self) -> None:
        # placeholder for pre-indexing steps
        print("Graph indexing started. No pre-indexing steps defined.")

    async def index_done_callback(self) -> None:
        # persist graph
        self.write_nx_graph(self._graph, self._graph_file)

    # --------------- Basic Queries ---------------
    async def check_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)
    
    async def check_edge(self, edge_source_id: str, edge_target_id: str) -> bool:
        return self._graph.has_edge(edge_source_id, edge_target_id)
    
    async def get_node_data_by_id(self, node_id: str) -> Union[dict[str, dict], None]:
        # logger.info(f"Existing or not: {node_id} in graph: {self._graph.has_node(node_id)}")
        # if self._graph.has_node(node_id):
        #     logger.info(f"{node_id} data: {self._graph.nodes.get(node_id)}")
        return self._graph.nodes.get(node_id)
    
    async def get_edge_data_by_id(self, edge_source_id: str, edge_target_id: str) -> Union[dict[tuple, dict], None]:
        return self._graph.edges.get((edge_source_id, edge_target_id))
    
    async def get_all_edges_from_this_node(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        if not self._graph.has_node(source_node_id):
            return None
        return list(self._graph.edges(source_node_id))
    
    async def get_node_degree(self, node_id: str) -> Union[int, None]:
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else None
    
    async def get_edge_degree(self, edge_source_id: str, edge_target_id: str) -> Union[int, None]:
        return (
            self._graph.degree(edge_source_id) if self._graph.has_node(edge_source_id) else 0
        ) + (
            self._graph.degree(edge_target_id) if self._graph.has_node(edge_target_id) else 0
        )
    
    async def update_or_insert_node(self, node_id: str, node_data_part: dict) -> None:
        self._graph.add_node(node_id, **node_data_part)
    
    async def update_or_insert_edge(self, edge_source_id: str, edge_target_id: str, edge_data_part: dict[str, str]) -> None:
        self._graph.add_edge(edge_source_id, edge_target_id, **edge_data_part)

    # ------------- Clustering & Schema -------------
    async def clustering(self, algorithm: str) -> None:
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        await self._clustering_algorithms[algorithm]()

    async def _leiden_clustering(self) -> None:
        if self._graph.number_of_nodes() == 0 or self._graph.number_of_edges() == 0:
            logger.warning("Graph is empty or trivial. Skipping clustering.")
            return  


        # use only largest component for stable clustering
        subgraph = self.stable_largest_connected_component(self._graph)
        partitions = hierarchical_leiden(
            subgraph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )
        # annotate nodes with cluster info
        cluster_map: Dict[str, List[Dict[str, int]]] = defaultdict(list)
        for p in partitions:
            cluster_map[p.node].append({'level': p.level, 'cluster': p.cluster})
        # write back into node attributes
        for node_id, clist in cluster_map.items():
            self._graph.nodes[node_id]['clusters'] = json.dumps(clist)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """
        Return largest connected component, relabeled and stabilized for deterministic ordering.
        """
        if graph.number_of_nodes() == 0:
            return nx.Graph()  # 返回空图避免报错
        from graspologic.utils import largest_connected_component
        sub = largest_connected_component(graph.copy())  # type: ignore
        # normalize node labels
        mapping = {n: html.unescape(n.upper().strip()) for n in sub.nodes()}  # type: ignore
        sub = nx.relabel_nodes(sub, mapping)
        return GraphDatabaseNetworkX._stabilize_graph(sub)  # type: ignore

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """
        Sort nodes and edges for deterministic GraphML output.
        """
        fixed = nx.DiGraph() if graph.is_directed() else nx.Graph()
        # sort nodes
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        fixed.add_nodes_from(sorted_nodes)
        # sort edges
        edges = list(graph.edges(data=True))
        if not graph.is_directed():
            # ensure undirected edges have consistent ordering
            edges = [
                (min(u, v), max(u, v), data)
                for u, v, data in edges
            ]
        edges = sorted(edges, key=lambda x: f"{x[0]}->{x[1]}")
        fixed.add_edges_from(edges)
        return fixed

    async def community_schema(self) -> Dict[str, SingleCommunityFormat_NoReport]:
        """
        Build hierarchical community schema from node 'clusters' attributes.
        """
        # gather info
        info: Dict[str, Any] = defaultdict(lambda: {
            'level': None,
            'title': None,
            'nodes': set(),
            'edges': set(),
            'chunk_ids': set(),
        })
        levels: Dict[int, set] = defaultdict(set)
        max_chunks = 0

        for node_id, data in self._graph.nodes(data=True):
            raw = data.get('clusters')
            if not raw:
                continue
            clusters = json.loads(raw)
            node_edges = self._graph.edges(node_id)
            src_chunks = set(data.get('source_chunk_ids', '').split(GRAPH_FIELD_SEP))

            for c in clusters:
                lvl, cid = c['level'], str(c['cluster'])
                levels[lvl].add(cid)

                rec = info[cid]
                rec['level'] = lvl
                rec['title'] = f"Cluster {cid}"
                rec['nodes'].add(node_id)
                for u, v in node_edges:
                    rec['edges'].add((min(u, v), max(u, v)))
                rec['chunk_ids'].update(src_chunks)
                max_chunks = max(max_chunks, len(rec['chunk_ids']))

        # build parent-child relations
        sorted_lv = sorted(levels)
        for i in range(len(sorted_lv) - 1):
            for pid in levels[sorted_lv[i]]:
                children = [
                    cid for cid in levels[sorted_lv[i+1]]
                    if info[cid]['nodes'].issubset(info[pid]['nodes'])
                ]
                info[pid]['sub_communities'] = children

        # finalize schema
        schema: Dict[str, SingleCommunityFormat_NoReport] = {}
        for cid, rec in info.items():
            schema[cid] = {
                'level': rec['level'],
                'title': rec['title'],
                'nodes': list(rec['nodes']),
                'edges': [list(e) for e in rec['edges']],
                'chunk_ids': list(rec['chunk_ids']),
                'occurrence': len(rec['chunk_ids']) / max_chunks if max_chunks else 0.0,
                'sub_communities': rec.get('sub_communities', []),
            }
        return schema

