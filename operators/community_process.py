# description是单纯的node或者edge的数据集合，Description -> LLM -> Report
from database.BaseClasses import BaseGraphStorage, BaseKVStorage
from database.data_format_specification.community_format import SingleCommunityFormat_NoReport, Community_withReport
from Use_LLM.prompt import PROMPTS



from operators.utils import logger, clean_think_tags, get_answer_from_qwen, truncate_list_by_token_size, list_of_list_to_csv, encode_string


import asyncio
from typing import List, Dict, Tuple, Set, Optional, Union

# 从低层次(2, 1, 0)循环level，level里面for每一个community，生成完整格式数据，调用_form_single_community_report -> generate_report_for_all_levels
# 每一个level生成完整格式，发给LLM自然需要足够多的information，这里面来自于社区的（nodes, entities, subcommunities的信息）-> _form_single_community_report
# _form_single_community_report调用_collect_description_for_single_community，返回三段数据{ 1. 有关nodes的描述, 2. 有关edges的描述, 3. 有关sub-communities的描述 }
# 如果需要子社区信息，_collect_description_for_single_community会调用_collect_information_from_sub_communities，收集返回一个子社区信息的4元组


async def generate_report_for_all_levels(
    community_instance: BaseKVStorage,
    graph_instance: BaseGraphStorage,
    config: dict,
):
    #     special_community_report_llm_kwargs: dict = field(
    #     default_factory=lambda: {"response_format": {"type": "json_object"}}
    # )
    
    # llm_extra_kwargs = config["special_community_report_llm_kwargs"]
    # use_llm_func: callable = config["best_model_func"]
    use_string_json_convert_func: callable = config[
        "string_json_convert_func"
    ]
    
    communities_schema = await graph_instance.community_schema()
    # cluster_keys, cluster_contents
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunityFormat_NoReport, processed_communities_in_lower_level: dict[str, Community_withReport]
    ):
        nonlocal already_processed
        
        description = await _collect_description_for_single_community(
            graph_instance,
            community,
            processed_communities_in_lower_level=processed_communities_in_lower_level,
            config=config,
        )
        community_report_prompt = PROMPTS["community_report"]
        prompt = community_report_prompt.format(input_text=description)

        llm_client = config["best_model_func"]
        response = await get_answer_from_qwen(client=llm_client, prompt=prompt)

        data = use_string_json_convert_func(response)
        already_processed += 1
        
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    # 0, 1, 2, 3用几个层次
    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_instance.update_or_insert(community_datas)
    
    
    
# 把子社区的report打包进社区里面
def _collect_information_from_sub_communities(
    current_community: SingleCommunityFormat_NoReport,
    processed_communities_in_lower_level: dict[str, Community_withReport],
    config: dict
) -> tuple[str, int]:
    # TODO
    # 里面有一个子社区sub_communities的字段
    valid_sub_communities = [
        processed_communities_in_lower_level[k] for k in current_community["sub_communities"] if k in processed_communities_in_lower_level
    ]
    
    if not valid_sub_communities:
        logger.warning(
            f"Community {current_community['title']} has no valid sub-communities, skipping report generation."
        )
        return "", 0, set(), set()
    # 根据出现频次排序？是这么排序的吗？
    valid_sub_communities = sorted(
        valid_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    
    # 合并他们的report_string，然后截断
    may_truncated_sub_communities = truncate_list_by_token_size(
        valid_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=config["max_tokensize_for_report_generation"],
    )

    sub_fields = ["id", "report", "rating", "importance"]
    description_from_communities = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_truncated_sub_communities)
        ]
    )
    
    from transformers import AutoTokenizer

    
    already_nodes = []
    already_edges = []
    for c in may_truncated_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        description_from_communities,
        len(encode_string(description_from_communities, config["model_name"])),
        set(already_nodes),
        set(already_edges),
    )


# class CommunitySchema(SingleCommunitySchema):
#     report_string: str
#     report_json: dict
async def _collect_description_for_single_community(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunityFormat_NoReport,
    processed_communities_in_lower_level: dict[str, Community_withReport] = {},
    config: dict = {},
) -> str:
    
    
    # 高层次可能会重复提取低层次的nodes和edges，可以稍微改进一下，用一个cache来存储已经提取过的节点和边
    
    # 先把community的nodes和edges的排序 
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    #　依次从graph中提取出来，合并
    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node_data_by_id(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge_data_by_id(src, tgt) for src, tgt in edges_in_order]
    )
    
    # 确立fileds 
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    
    # 列个关于nodes的表
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.get_node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    # 把nodes根据[-1]的属性，即是degree排序
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    # 截断nodes，防止过长
    
    max_token_size = config["max_tokensize_for_report_generation"] // 2
    
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size//2
    )
    
    # 构建一个edges的列表
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.get_edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    
    # 调整一下顺序
    # 根据degree排序，截断
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size//2
    )

    # 判断nodes和edges的和截断前相比谁更长
    truncated_or_not = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    
    # 当确实截断了，而且有子社区，而且有已经存在的子社区的报告
    need_to_use_sub_communities = (
        truncated_or_not and len(community["sub_communities"]) and len(processed_communities_in_lower_level)
    )
    # 用户可以在全局配置里强制要求无论如何都用子社区。
    force_to_use_sub_communities = config.get(
        "use_sub_communities_or_not", False
    )
    
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        
            # _pack_single_community_by_sub_communities返回这个
            #     return (
            #           sub_communities_describe,
            #           len(encode_string_by_tiktoken(sub_communities_describe)),
            #           set(already_nodes),
            #           set(already_edges),
            #           )
        report_describe, report_size, nodes_contained_in_sub_communities, edges_contained_in_sub_communities = (
            _collect_information_from_sub_communities(
                community, processed_communities_in_lower_level, config
            )
        )
        
        # nodes_list_data自己的， contain_nodes是子社区得来的
        # 根据哪些节点/边已经由子社区报告覆盖，把它们分成“已包含”和“未包含”两部分
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in nodes_contained_in_sub_communities # 子社区会截断，所以有的不包含
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in nodes_contained_in_sub_communities
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in edges_contained_in_sub_communities
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in edges_contained_in_sub_communities
        ]
        
        # 用剩余的 token 配额（max_token_size - report_size），再次对节点和边列表做截断，保证总长度不超限。
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2, # 余额一半给给nodes的
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2, # 余额一半给edge
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"