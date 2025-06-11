from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, TypedDict

schema_results = dict[str, dict[str, Any]]
# {'17': {'level': 0, 'title': 'Cluster 17', 'nodes': ['THE GHOST OF CHRISTMAS YET TO COME', "THE WRETCHED MAN'S DEATH"], 'edges': [['THE GHOST OF CHRISTMAS YET TO COME', "THE WRETCHED MAN'S DEATH"], ['SCROOGE', 'THE GHOST OF CHRISTMAS YET TO COME']], 'chunk_ids': [''], 'occurrence': 1.0, 'sub_communities': []}}}

SingleCommunityFormat_NoReport = TypedDict(
    "Community_Format_without_Report",
    {
        "level": int,
        "title": str,
        "edges": list[tuple[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)

@dataclass
class Community_withReport(SingleCommunityFormat_NoReport):
    """包含报告的社区数据格式，继承基本社区格式"""
    report_string: str
    report_json: Dict[str, Any]