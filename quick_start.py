from my_graphrag.graph_rag import GraphRAG
from database.implemented_storage_classes.graph_database_NX import GraphDatabaseNetworkX
from operators.utils import get_serializable_config
from database.implemented_storage_classes.key_string_value_JSON_database import KeyStringValueJsonDatabase
from my_graphrag.base_config import BaseConfig, QueryParam
import asyncio

graph_rag_instance = GraphRAG()


with open("./Christmas Carol.txt", encoding="utf-8") as file:
    text = file.read()
    graph_rag_instance.insert(text)
    
    
answer = graph_rag_instance.query(query="What is the main theme of the story?")
print("Answer:\n", answer)