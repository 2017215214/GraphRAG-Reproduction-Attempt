from typing import TypedDict

        # source_node_id = source_node,
        # target_node_id = target_node,
        # edge_description = edge_description,
        # edge_source_chunk = edge_source_chunk,
        # weight = weight

MergedEdgeFormat = TypedDict(
    "MergedEdgeFormat",
    {
        "source_node_id": str,
        "target_node_id": str,
        "edge_data": dict, # 包含weight，description，source_chunk_ids，order
        
    }
)