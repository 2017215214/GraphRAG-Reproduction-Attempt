from dataclasses import dataclass, field
from database.BaseClasses import BaseVectorStorage
from nano_vectordb import NanoVectorDB
import numpy as np
import os
from operators.utils import logger
import asyncio



# 存储形如dict[str, dict]
@dataclass
class VectorDatabase(BaseVectorStorage):
    better_than_threshold: float = 0.2
    
    # 借用NanoVectorDB来完成实例新对象的核心部分
    def __post_init__(self):
        self._client_file_path = os.path.join(self.global_config["working_dir"], f"vector_database_{self.name_space}.json")
        self._max_batch_size = self.global_config["embedding_batch_size"] or 16
        self._client_object = NanoVectorDB(
            self.global_config["embedding_dim"], storage_file=self._client_file_path
            )
        self.better_than_threshold = self.global_config["query_better_than_threshold"] or self.better_than_threshold
            
        
    async def index_start_callback(self):
        # print(f"Indexing started for {self.name_space} storage.")
        pass
            
    async def index_done_callback(self):
        # print(f"Indexing done for {self.name_space} storage.")  
        self._client_object.save()    
    
    # text → chunk 阶段，做的工作通常是：文本清洗（去掉无关字符、HTML、换行等）
    # 切分成适合大小的 chunk（比如按句、段、token 数限制），加上 metadata（如来源标题、段落编号、页码等）
    # text_db -> chunk_db: str -> token, chunk_db -> vector_db: token -> embedding
    
    # 优化：支持 Relation 的双向查询（边两端的 node）
    
    # inserted_data因为是dict数据，所以本身可能也是多条的
    async def update_or_insert(self, inserted_data: dict[str, dict]) -> list:
        if any("content" not in v for v in inserted_data.values()):
            raise ValueError("All inserted data must have 'content' field.")

        logger.info(f"Inserting {len(inserted_data)} vectors to {self.name_space}")
        if not len(inserted_data):
            logger.warning("You insert an empty data to vector DB")
            return []

        # 多个dict_element 展开成 -> list[dict]，不要用str_id来领头了
        # **，表示字典解包，表示和"__id__": k 一起”合并成一个新字典。
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            }
            for k, v in inserted_data.items()
        ]
        
        print(f"list_data = {list_data[0]}")
        # 编码生成embedding
        # v是一个字典，里面有个content，后面是tokens的list
        tokens_content = [x["content"] for x in inserted_data.values()]
        batches = [
            tokens_content[i: i + self._max_batch_size] for i in range(0, len(tokens_content), self._max_batch_size)
            ]
        embedding_func = self.global_config["embedding_func"]
        # async def async_encode(embedding_func, batch):
        #     loop = asyncio.get_event_loop()
        #     return await loop.run_in_executor(None, embedding_func.encode, batch)

        # embeddings_list = await asyncio.gather(
        #     *[async_encode(embedding_func, batch) for batch in batches]
        # )
        # 需要的话再封装成async
        embeddings_list = []
        for batch in batches:
            embeddings = embedding_func.encode(batch)
            embeddings_list.append(embeddings)
        
        embeddings = np.concatenate(embeddings_list)
        for i, data in enumerate(list_data):
            data["__vector__"] = embeddings[i]
            # print(f"data = {data['__vector__'][:10]}\n type: {type(data['__vector__'])}")
            
        # 注意选用embedding模型后要选择相应的embeddings唯独，在base_config.py里面设置好embedding_dim
        results = self._client_object.upsert(list_data)
        # 这results只插入keys，具体数据在self._client_object["_NanoVectorDB__storage"]里面查看
        # logger.info(f"Inserted {len(list_data)} vectors to {self.name_space}")
        return results
        
    # 假设每一个元素插入nano-vectordb里面后如下：
    # {
    #   "__id__": "chunk-012",
    #   "content": [0.1, 0.2, 0.3, 0.4],
    #   "__vector__": [embeddings],
    #   "__metrics__": 0.85
    # }
        
    async def query(self, query: str, top_k: int) -> list[dict]:
        # 确保传入的是字符串，而不是列表
        if isinstance(query, list):
            query = query[0] if query else ""
        
        # 直接使用 encode 方法，SentenceTransformer 的标准用法
        vec = self.embedding_func.encode(query)
        
        import numpy as np
        arr = np.array(vec)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        
        results = self._client_object.query(arr, top_k=top_k)
        return results