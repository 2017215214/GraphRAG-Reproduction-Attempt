from typing import TypedDict

        # entity_name = entity_name,
        # entity_type = entity_type,
        # entity_description = entity_description,
        # entity_source_chunk = entity_source_chunk_key

NodeFormat = TypedDict(
    "NodeFormat",
    {
        "type": str,
        "description": str, 
        "source_chunk_id": str   
    }
)

MergedNodeFormat = TypedDict(
    "MergedNodeFormat",
    {
        "type": str, # 用分隔符分开
        "description": str, # 用分隔符分开
        "source_chunk_ids": str# 用分隔符分开
    }
)