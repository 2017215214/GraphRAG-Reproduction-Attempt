from typing import Generic, TypeVar, Union, Optional
from dataclasses import dataclass, field
from operators.utils import EmbeddingFunction
from database.data_format_specification.community_format import SingleCommunityFormat_NoReport



T = TypeVar("T")

@dataclass
class StorageNameSpace:
    """
    Base class for all storage implementations.
    
    Attributes:
        name_space (str): Unique identifier for storage instance
        global_config (BaseConfig): Global configuration
    """
    name_space: str
    global_config: dict # 使用dict而不是BaseConfig类型
    
    # 当“索引构建任务完成”时，调用这个回调，把内存中的 _data 写入到磁盘 JSON 文件中，实现持久化保存。
    # 如果不传self就像静态方法，后续实现的时候也不能用self的attribute和method
    async def index_start_callback(self) -> None:
        # pass意思是「这个函数暂时什么都不做」，但不会报错。你可以直接调用它，它什么都不会发生。可以有子类覆盖，这是一种设计模式。
        # raise NotImplementedError()意思是「这个方法必须由子类实现」。如果你忘了实现、直接调用就会报错。必须做点什么。
        raise NotImplementedError
    
    async def index_done_callback(self) -> None:
        raise NotImplementedError
    
    async def query_done_callback(self) -> None:
        raise NotImplementedError
    

# 考虑搭配BaseKVStorage后续会有好几种存储格式，对T格式实现，接口返回T相关的格式
# 继承一下Generic[T]，返回的时候就好些了，当个占位符
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    
    async def get_all_keys(self) -> list[str]:
        raise NotImplementedError
    
    async def get_data_by_id(self, id: str) -> Union[T, None]:
        # 建议添加
        if not isinstance(id, str):
            raise ValueError("ID must be string type")
        raise NotImplementedError
    
    async def get_data_by_ids(self, ids: list[str], fields: set[str]) -> list[Union[T], None]:
        raise NotImplementedError
    
    async def get_keys_to_be_inserted(self, external_keys_to_be_inserted: list[str]) -> set[str]:
        raise NotImplementedError
    
    async def update_or_insert(self, data: dict[str, T]) -> None:
        raise NotImplementedError
    
    async def clear_all(self) -> None:
        raise NotImplementedError
    
# chunk[content] -> embedding
@dataclass
class BaseVectorStorage(Generic[T], StorageNameSpace):
    
    embedding_func: EmbeddingFunction
    # 除了content，可能有一些其他的field你想加入
    # 如果直接写 = set()会导致所有实例共用一个set，就不能互相隔离
    # 这样写会每次实例化的时候创建一个新的字段集合，默认是空集合
    
    # 不是所有字段都需要存储，meta_fileds制定哪些需要存储
    meta_fields: set[str] = field(default_factory=set)

    
    # 返回top_k个数据，每个数据如{data_id: {field_0: xxx, field_1: xxx}}
    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError
    
    async def update_or_insert(self, data: dict[str, dict]):
        raise NotImplementedError
    
@dataclass
class BaseGraphStorage(StorageNameSpace):
    
    async def check_node(self, node_id: str) -> bool:
        raise NotImplementedError
    
    async def check_edge(self, edge_source_id: str, edge_target_id: str) -> bool:
        raise NotImplementedError
    
    async def get_node_data_by_id(self, node_id: str) -> Union[dict[str, dict], None]:
        raise NotImplementedError
    
    async def get_edge_data_by_id(self, edge_source_id: str, edge_target_id: str) -> Union[dict[tuple, dict], None]:
        raise NotImplementedError
    
    async def get_nodes(self, nodes_id: list[str]) -> list[Union[dict[str, dict], None]]:
        raise NotImplementedError
    
    # 不需要获取data，只需要名字，注意返回格式不是list[Union[tuple[str, str], None]]，因为不存在的只可能是这一个node
    async def get_all_edges_from_this_node(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError
    
    async def get_node_degree(self, node_id: str) -> Union[int, None]:
        raise NotImplementedError
    
    async def get_edge_degree(self, edge_source_id: str, edge_target_id: str) -> Union[int, None]:
        raise NotImplementedError
    
    # 这里我们把node_data = {id: str, data: {field_0: xxx, field_1: xxx}}只是拆成两部分
    # node_id是一个str， 后面的提取出来的data自然是key: value格式
    async def update_or_insert_node(self, node_id: str, node_data_part: dict[str, str]) -> None:
        raise NotImplementedError
    
    async def update_or_insert_edge(self, edge_source_id: str, edge_target_id: str, edge_data_part: dict[str, str]):
        raise NotImplementedError
    
    async def clustering(self, algorithm: str) -> None:
        raise NotImplementedError
    
    async def generate_initial_communities(self) -> dict[str, SingleCommunityFormat_NoReport]:
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunityFormat_NoReport]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError
    
    # async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
    #     raise NotImplementedError("Node embedding is not used in nano-graphrag.")