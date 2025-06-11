from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, Type, Callable, List, Dict, Literal
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from Use_LLM.embedding_api import QwenEmbeddingAPI
from Use_LLM.chat_api import QwenChatAPI

from database.BaseClasses import (
    BaseKVStorage, 
    BaseVectorStorage, 
    BaseGraphStorage
)
from database.implemented_storage_classes.graph_database_NX import GraphDatabaseNetworkX
from database.implemented_storage_classes.vector_database_NanoVectorDB import VectorDatabase
from database.implemented_storage_classes.key_string_value_JSON_database import KeyStringValueJsonDatabase
from others.use_qwen import qwen_llm
from others.qwen_embedding import qwen_embedding
from operators.utils import EmbeddingFunction
from Use_LLM.chat_api import QwenChatAPI


@dataclass
class BaseConfig:
    # 基础配置
    working_dir: str = field(
        default_factory=lambda: f"./Cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # 文本处理配置
    chunk_func: Optional[Callable[
        [
            List[List[int]],  # tokenized documents
            List[str],        # doc_ids
            tiktoken.Encoding,
            int,             # overlap_token_size
            int,             # chunk_token_size
        ],
        Dict[str, Dict[str, Union[str, int]]]  
    ]] = None
    chunk_token_size: int = 1200
    overlap_token_size: int = 100
    
    # 向量配置
    embedding_func: SentenceTransformer = field(default_factory=lambda: SentenceTransformer("BAAI/bge-small-zh-v1.5")
)
    embedding_max_async_thread_num: int = 16
    embedding_batch_size: int = 16 
    query_better_than_threshold: float = 0.2
    waitting_time: float = 0.0001
    embedding_dim: int = 512
    
    # 图配置
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    cluster_partition_algorithm: str = "leiden"
    
    # 抽取配置
    extract_func: Optional[Callable] = None
    max_gleanning_time: int = 1
    
    # LLM配置
    model_name: str = "Qwen/Qwen3-8B"
    best_model_func: callable = field(default_factory=lambda: QwenChatAPI(model="qwen3", api_url="http://localhost:8001"))
    enable_llm_cache: bool = True
    use_llm_or_not: bool = True
    llm_window_size: int = 16384  # Qwen3-8B的上下文窗口大小
    summary_tokens_max_length: int = 512
    
    # 查询配置
    enable_local_query: bool = False
    enable_naive_query: bool = True
    enable_global_query: bool = True
    
    # communities配置
    string_json_convert_func: callable = None
    max_tokensize_for_report_generation: int = 16384 # 32768
    use_sub_communities_or_not: bool = False
    
    
    
    def __post_init__(self):
        # 确保工作目录存在
        import os
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            print(f"Created working directory: {self.working_dir}")
        
        # 设置默认函数
        if self.chunk_func is None:
            from operators.text_chunk_process import chunking_by_token_size
            self.chunk_func = chunking_by_token_size
            
        if self.extract_func is None:
            from operators.element_extraction import extract_extities_and_reations
            self.extract_func = extract_extities_and_reations
            
        if self.string_json_convert_func is None:
            from operators.utils import convert_response_to_json
            self.string_json_convert_func = convert_response_to_json
            

@dataclass
class QueryParam:
    # 这些参数也配置了如下面Community Schema的一些参数
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    deepest_level: int = 2
    top_k: int = 20
    
    # naive search
    naive_max_token_for_text_unit = 12000
    
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384 # 32768, 8192
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

