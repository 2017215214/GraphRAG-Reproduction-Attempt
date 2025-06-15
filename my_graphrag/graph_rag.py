from dataclasses import dataclass, asdict, field
from typing import cast, Type, Dict, Union, Optional
from database.implemented_storage_classes.key_string_value_JSON_database import KeyStringValueJsonDatabase
from database.implemented_storage_classes.vector_database_NanoVectorDB import VectorDatabase
from database.implemented_storage_classes.graph_database_NX import GraphDatabaseNetworkX
from database.BaseClasses import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
import asyncio
import time
import os


from my_graphrag.base_config import BaseConfig, QueryParam


from database.BaseClasses import StorageNameSpace

from operators.utils import (
    get_event_loop,
    logger,
    limit_async_func_call,
    get_serializable_config,
)

from operators.text_chunk_process import (
    process_docs_to_new_fromat,
    split_texts_into_chunks,
)

from operators.community_process import (
    generate_report_for_all_levels,
)

from operators.global_query import global_query
from operators.naive_query import naive_query


# 注意利用AI添加注释，还有输出，作为可视化
# 解决think输出，prompt的添加，get_answer_from_qwen的输出等问题，prompt添加在truncate_prompt里面可能会比较好？？？
# 再看一下global_query里面的145行的断点输出，很奇怪，老是输出think
# Looking at the first entry, "importance": 0.030303030303030304，太低导致思考
# 调低temperature，结果啥都输出不了，一直在confusing
# 注意每个prompt会不会再送给llm的时候，由于过长，都会被送给llm的时候被截断
# 统一修改get_answer_from_wen，引入model参数，直接在config里面修改完就可以了
# 合并baseconfig里的参数，比如llm_window_size就是prompt_max_token_size
# 添加cache
# 统一命名
@dataclass
class GraphRAG:
    # 这样每个实例都会获得自己的 BaseConfig 实例
    config: BaseConfig = field(default_factory=BaseConfig)
    
    # 存储类配置
    kv_dict_database_class: Type[BaseKVStorage] = KeyStringValueJsonDatabase
    vector_database_class: Type[BaseVectorStorage] = VectorDatabase
    graph_database_class: Type[BaseGraphStorage] = GraphDatabaseNetworkX
    
    
    def __post_init__(self):
        
        cfg = get_serializable_config(self.config)
        
        if not os.path.exists(cfg["working_dir"]):
            os.makedirs(cfg["working_dir"])
            logger.info(f"Creating directory: {cfg['working_dir']}")
          
        # choose LLM 
        
        
        # 实例化
        self.entity_relation_graph = self.graph_database_class(
            # 各base类型里，传入的就是global_config，自己会提取字段，以后若需要别的字段，扩展也更方便
            name_space="entity_relation", global_config=cfg
        )
        
        self.original_docs = self.kv_dict_database_class(
            name_space="original_docs", global_config=cfg
        )
        
        self.chunks = self.kv_dict_database_class(
            name_space="chunks", global_config=cfg
        )
        
        # 根据全局配置，包装出来的
        self.embedding_func_under_restriction = limit_async_func_call(func=cfg["embedding_func"], 
                                                    max_size=cfg["embedding_max_async_thread_num"], 
                                                    waitting_time=cfg["waitting_time"])
        
        # BaseConfig里的embedding_func和自己包装过后的self.embedding_func不一样
        
        # 简单的sentence_transformer向量查询通常不需要异步并发，别的另当别论。但是上面的包装会让小模型失去encode方法
        self.chunk_vectors = self.vector_database_class(
            name_space="chunk_vectors", global_config=cfg,
            embedding_func=cfg["embedding_func"],
            meta_fields={"entity_name"}
        ) if cfg["enable_naive_query"] else None
            
        
        self.entity_vetors = self.vector_database_class(
            name_space="entity_vectors",global_config=cfg,
            embedding_func=self.embedding_func_under_restriction
        ) if cfg["enable_naive_query"] else None
        
        self.community_reports = self.kv_dict_database_class(
            name_space="community_reports", global_config=cfg,
        )
        
        
    def insert(self, docs):
        loop = get_event_loop()
        return loop.run_until_complete(self.async_insert(docs))
    
    async def async_insert(self, docs):
        await self._insert_start()
        try:
            
            config = get_serializable_config(self.config)

            # -------------------text process------------------- #
            if isinstance(docs, str):
                docs = [docs]
            else:
                logger.info("Input is not string or string_string")
                return
            
            # text存储格式见文档3.1.1
            new_docs = process_docs_to_new_fromat(docs)
            # 不用await就只是一个coroutine对象
            keys_not_in_doc_database = await self.original_docs.get_keys_to_be_inserted(new_docs.keys())
            if len(keys_not_in_doc_database) == 0:
                logger.info("All docs are already in storage.")
                return
            # 边处理数据，边插入数据库，会造成流水线堵塞和上下文切换等问题，等到最后统一insert
            new_docs_to_be_inserted = {k: v for k, v in new_docs.items() if k in keys_not_in_doc_database}
            
            
            # -------------------get chunks------------------- #
            new_chunks = split_texts_into_chunks(texts=new_docs_to_be_inserted, 
                                                 config=config)
            # 注意，调用async方法要用await
            keys_not_in_chunks_databse = await self.chunks.get_keys_to_be_inserted(new_chunks.keys())
            if len(keys_not_in_chunks_databse) == 0:
                logger.info("There's not new chunks which needs to insert into chunks database.")
                return
            # 等到最后一起插入
            chunks_to_be_inserted = {k: v for k, v in new_chunks.items() if k in keys_not_in_chunks_databse}
            # logger.info(f"{len(keys_not_in_chunks_databse)} chunks will be inserted.")
            
            
            # -------------------community and graph process------------------- #
            # no incremental operations temporaroly, just drop all
            await self.community_reports.clear_all()
            # logger.info("Community reports cleared.")
            # logger.info(("Entities and Relationship are being Extracted..."))
            # extract_func = config["extract_func"]
            # print(extract_func)
            
            # start_time = time.time()
            temporary_entity_relation_graph = await config["extract_func"](chunks=chunks_to_be_inserted,
                                                                            config=config,
                                                                            entity_vector_database=self.entity_vetors,
                                                                            graph_instance=self.entity_relation_graph)
            # end_time = time.time()
            # logger.info(f"Entity and Relationship Extraction took {end_time - start_time:.2f} seconds.")
        
            if temporary_entity_relation_graph is None:
                logger.warning("No new entities found")
                return
            self.entity_relation_graph = temporary_entity_relation_graph
            
            
            logger.info("Generating Community Reports...")
            

            await self.entity_relation_graph.clustering(config['cluster_partition_algorithm'])
            
            
            await generate_report_for_all_levels(community_instance=self.community_reports,
                                                 graph_instance=self.entity_relation_graph,
                                                 config=config)
            logger.info("Community Reports generated successfully.")
            
            # -------------------insert data------------------- #
            # logger.info(f"Inserting {len(keys_not_in_doc_database)} into docs_databse.")
            await self.original_docs.update_or_insert(new_docs_to_be_inserted)
            # logger.info(f"Inserting {len(keys_not_in_chunks_databse)} into chunks_databse.")
            await self.chunks.update_or_insert(chunks_to_be_inserted)
            
            if config["enable_naive_query"]:
                # logger.info("Insert chunks for naive RAG")
                await self.chunk_vectors.update_or_insert(chunks_to_be_inserted)
        
        finally:
            await self._insert_done()
    
    
    async def _insert_start(self):
        start_tasks = []
        for i in [self.entity_relation_graph]:
            start_tasks.append(cast(StorageNameSpace, i).index_start_callback())
        await asyncio.gather(*start_tasks)
        
        
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            ##  KV
            self.original_docs,
            self.chunks,
            # self.llm_response_cache,
            self.community_reports,
            
            # Vectors
            self.entity_vetors,
            self.chunk_vectors,
            
            # Graph
            self.entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            # 插入结束后，通知每个存储“请提交/落盘/刷新索引”
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
        
    
    def query(self, query: str, query_para: QueryParam=QueryParam()):
        loop = get_event_loop()
        return loop.run_until_complete(self.async_query(query=query, query_para=query_para))
        
    
    async def async_query(self, query_para: QueryParam, query: str):
        
        config = get_serializable_config(self.config)
        ## 后面注意放到主循环里，然后删掉这一段
        
        if query_para.mode == 'naive' and not config["enable_naive_query"]:
            raise ValueError("Naive query is not enabled in the configuration.")
        
        if query_para.mode == 'naive':
            return await naive_query(query=query,
                                     chunk_database=self.chunks,
                                     chunk_vector_database=self.chunk_vectors,
                                     query_param=query_para,
                                     config=config)
            
        
        elif query_para.mode == 'global':
            return await global_query(
                graph_instance=self.entity_relation_graph,
                query=query,
                query_param=query_para,
                community_instance=self.community_reports,
                config=config
            )
    
    
            
            
    def test_query(self, query_para: QueryParam, query: str, config: dict, chunk_database: BaseKVStorage, chunk_vector_database: BaseVectorStorage):
        loop = get_event_loop()
        config = get_serializable_config(self.config)
        return loop.run_until_complete(self.async_query(query=query, 
                                                        chunk_database=chunk_database,
                                                        chunk_vector_database=chunk_vector_database,
                                                        query_param=query_para,
                                                        config=config))
        
    
    async def test_async_query(self, query_para: QueryParam, query: str, chunk_database: BaseKVStorage, chunk_vector_database: BaseVectorStorage, config: dict):        
        if query_para.mode == 'naive' and not self.config["enable_naive_query"]:
            raise ValueError("Naive query is not enabled in the configuration.")
        
        if query_para.mode == 'naive':
            return await naive_query(query=query,
                                     chunk_database=chunk_database,
                                     chunk_vector_database=chunk_vector_database,
                                     query_param=query_para,
                                     config=config)

        
        
        
    # def generate_communities(self, graph_instance: BaseGraphStorage, community_instance: BaseKVStorage, config: dict):
    #     loop = get_event_loop()
    #     return loop.run_until_complete(self.async_generate_communities(graph_instance=graph_instance, community_instance=community_instance, config=config))
    
    # async def async_generate_communities(self, graph_instance: BaseGraphStorage, community_instance: BaseKVStorage, config: dict):

    #     await graph_instance.clustering("leiden")
    #     await graph_instance.community_schema()
    #     await generate_report_for_all_levels(graph_instance=graph_instance, community_instance=community_instance, config=config)