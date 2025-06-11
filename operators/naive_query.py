from database.BaseClasses import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from my_graphrag.base_config import QueryParam
from Use_LLM.prompt import PROMPTS
from operators.utils import logger, truncate_list_by_token_size, truncate_prompt, get_answer_from_qwen

async def naive_query(query: str, 
                      chunk_database: BaseKVStorage,
                      chunk_vector_database: BaseVectorStorage,
                      query_param: QueryParam,
                      config: dict
                      ) -> str:
    
    selected_chunks = await chunk_vector_database.query(query=query, top_k=query_param.top_k)
    if not len(selected_chunks):
        logger.warning("No chunks found for the query.")
        return "No relevant information found."
    
    chunk_ids = [c["__id__"] for c in selected_chunks]
    chunk_datas = await chunk_database.get_data_by_ids(chunk_ids)
    truncated_chunk_datas = truncate_list_by_token_size(list_data=chunk_datas,
                                                        key=lambda x: x["content"],
                                                        max_token_size=query_param.naive_max_token_for_text_unit)
    
    logger.info(f"Selected {len(truncated_chunk_datas)} chunks for the query.")
    
    section_texts = "------New Chunk------\n".join(c["content"] for c in truncated_chunk_datas)
    if query_param.only_need_context:
        return section_texts
    
    prompt = PROMPTS["naive_rag_response"].format(content_data=section_texts, 
                                                  response_type=query_param.response_type
                                            )
    prompt= truncate_prompt(prompt=prompt,
                            query=query,
                            model_name=config["model_name"],
                            max_token_size=config["llm_window_size"])
    llm_client = config["best_model_func"]
    response = await get_answer_from_qwen(prompt=prompt, client=llm_client)
    
    return response
    
    
    