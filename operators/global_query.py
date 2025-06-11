from database.BaseClasses import BaseGraphStorage, BaseKVStorage

from database.data_format_specification.community_format import Community_withReport, SingleCommunityFormat_NoReport
from my_graphrag.base_config import QueryParam
from Use_LLM.prompt import PROMPTS
import asyncio

from operators.utils import logger, truncate_list_by_token_size, list_of_list_to_csv, get_answer_from_qwen, truncate_prompt

# 因为会加入query，所以要动态调整上下文限制
async def global_query(graph_instance: BaseGraphStorage, query: str, query_param: QueryParam, community_instance: BaseKVStorage, config: dict) -> str:
    
    community_schema = await graph_instance.community_schema()
    commmunities_under_designated_level = {
        k: v for k, v in community_schema.items() if v['level'] <= query_param.deepest_level
    }
    if len(commmunities_under_designated_level) < 1:
        raise ValueError("No communities found at the specified level.")
    
    
    llm_clinet = config.get("best_model_func", None)
    if llm_clinet is None:
        raise ValueError("LLM client is not configured in the provided config.")
    
    # 应该是根据社区内每个节点的degree的和，来做排序指标
    # sort and truncate
    sorted_communities = sorted(commmunities_under_designated_level.items(), key=lambda x: x[1]['occurrence'], reverse=True)
    truncated_communities = sorted_communities[:query_param.global_max_consider_community]
    
    
    community_datas = await community_instance.get_data_by_ids(
        c[0] for c in truncated_communities
        )
    community_datas = [c for c in community_datas if c is not None]
    
    # 可以加一个每一个description对应哪一个community，方便后续追踪
    used_communities: list = [
        c for c in community_datas if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    logger.info(f"Revtrieved {len(used_communities)} communities")
    
    # divide communities into different groups, and for each group generate multiple points
    community_group_points = await _map_communities_into_groups_and_get_points(used_communities=used_communities, config=config, query_param=query_param)
    # [
    #   { group_0: [point_0: {descrip, score}, point_1: {descrip, score}, ...] },
    #   { group_1: [point_0: {descrip, score}, point_1: {descrip, score}, ...]},
    # ]       
    
    cleaned_group_points = []
    for i, group in enumerate(community_group_points):
        for point in group:
            if "description" not in point: continue
            cleaned_group_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    cleaned_group_points = [g for g in cleaned_group_points if g["score"] > 0]
            
    if not len(cleaned_group_points):
        logger.warning("No valid points found after cleaning.")
        return "No valid points found."
    
    cleaned_group_points = sorted(
        cleaned_group_points, key=lambda x: x["score"], reverse=True
    )
     
    used_group_points = truncate_list_by_token_size(
        list_data=cleaned_group_points,
        key= lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report
    )
    
    points_context = []       
    for p in used_group_points:
        points_context.append(f"""----Analyst {p['analyst']}----Importance Score: {p['score']}{p['answer']}""")
    
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    
    prompt = PROMPTS["global_reduce_rag_response"]
    prompt = prompt.format(context_data=points_context, report_data=points_context, response_type=query_param.response_type)
    
    prompt = truncate_prompt(prompt=prompt,
                             query=query, 
                             model_name=config["model_name"], 
                             max_token_size=query_param.global_max_token_for_community_report
                             )
    
    response = await get_answer_from_qwen(client=config["best_model_func"], prompt=prompt, query=query)
    
    return response
    
    
    
    
    
    
    
    

async def _map_communities_into_groups_and_get_points(used_communities: list[Community_withReport], config: dict, query_param: QueryParam) -> list[dict]:
    # divide groups by sliding window of token size
    community_groups = []
    max_token_size = query_param.global_max_token_for_community_report
    
    while(len(used_communities)):
        community_for_this_group = truncate_list_by_token_size(
            list_data=used_communities,
            key=lambda x: x["report_string"],
            max_token_size=max_token_size
        )
        community_groups.append(community_for_this_group)
        used_communities = used_communities[len(community_for_this_group):]
        
    logger.info(f"Divided communities into {len(community_groups)} groups.")
    
    # for each group, generate points
    async def _generate_points_for_each_group(this_community_group: list[Community_withReport],
                                             client) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        # 提取出community_truncated_datas关键信息，为发送LLM获得更大社区群组答案做准备
        for i, c in enumerate(this_community_group):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        # 整理成context的csv格式
        community_context = list_of_list_to_csv(communities_section_list)
        prompt = PROMPTS["global_map_rag_points"]
        prompt = prompt.format(context_data=community_context)
        
        response = await get_answer_from_qwen(client=client, prompt=prompt)
        print(f"Group Response: {response}")
        if response is None:
            logger.error("Failed to get response from LLM.")
            return {}
        response = config["string_json_convert_func"](response)
        key_points = response.get("points", [])
        
        return key_points
    
    responses =await asyncio.gather(
        *[
          _generate_points_for_each_group(this_community_group=c, client=config["best_model_func"])
          for c in community_groups  
        ]
    )
    return responses
        