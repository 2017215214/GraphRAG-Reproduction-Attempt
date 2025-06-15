from my_graphrag.graph_rag import GraphRAG
from database.implemented_storage_classes.graph_database_NX import GraphDatabaseNetworkX
from operators.utils import get_serializable_config, logger
from database.implemented_storage_classes.key_string_value_JSON_database import KeyStringValueJsonDatabase
from my_graphrag.base_config import BaseConfig, QueryParam
import asyncio
import time

graph_rag_instance = GraphRAG()

with open("./Christmas Carol.txt", encoding="utf-8") as file:
    text = file.read()
    
time_0 = time.time()
graph_rag_instance.insert(text)

time_1 = time.time()
logger.info(f"Insert time: {time_1 - time_0} seconds")  

answer = graph_rag_instance.query(query="What is the main theme of the story?", query_para=QueryParam(mode="naive"))
print("Naive Answer:\n", answer)

time_2 = time.time() 
logger.info(f"Naive query time: {time_2 - time_1} seconds") 

answer = graph_rag_instance.query(query="What is the main theme of the story?", query_para=QueryParam(mode="global", deepest_level=2))
print("Level 2 Answer:\n", answer)

time_3 = time.time()  
logger.info(f"Level 2 query time: {time_3 - time_2} seconds")

answer = graph_rag_instance.query(query="What is the main theme of the story?", query_para=QueryParam(mode="global", deepest_level=1))
print("Level 1 Answer:\n", answer)

time_4 = time.time() 
logger.info(f"Level 1 query time: {time_4 - time_3} seconds") 

answer = graph_rag_instance.query(query="What is the main theme of the story?", query_para=QueryParam(mode="global", deepest_level=0))
print("Level 0 Answer:\n", answer)

time_5 = time.time()  
logger.info(f"Level 0 query time: {time_5 - time_4} seconds")