from typing import TypedDict
from dataclasses import dataclass

ChunkFormat = TypedDict("ChunkFormat", 
                        {
                            "tokens": int,
                            "content": str,
                            "original_doc_id": str,
                            "chunk_order_index": int,
                        },
                        )