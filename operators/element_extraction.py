from database.BaseClasses import BaseGraphStorage, BaseVectorStorage
from my_graphrag.base_config import BaseConfig
from Use_LLM.prompt import GRAPH_FIELD_SEP, PROMPTS
import asyncio
import re
from collections import defaultdict
from database.data_format_specification.chunk_format import ChunkFormat
from database.data_format_specification.node_format import MergedNodeFormat
from database.data_format_specification.edge_format import MergedEdgeFormat

from operators.utils import (
    logger,
    pack_user_ass_to_openai_messages,
    split_by_multimarkers,
    clean_str,
    is_float_regex,
    clean_think_tags,
    clean_records,
    compute_id_by_mdhash,
    extend_elements_to_list_if_not_exists,
    truncate_list,
    merge_string_fields_with_separator,
    get_answer_from_qwen,
    )


# 返回一部分nodes和edges，返回来list[nodes_dict], list[edges_dict]，允许重复
async def extract_extities_and_reations(chunks: dict[str, ChunkFormat], 
                                        config: dict, 
                                        entity_vector_database: BaseVectorStorage,
                                        graph_instance: BaseGraphStorage) -> tuple[dict[str, dict], dict[tuple[str, str], dict]] | None:
    
    entity_extract_prompt = PROMPTS["entity_extraction"]
    llm_client = config["best_model_func"]
    max_gleanning_time = config["max_gleanning_time"]
    
    try:
        results = await asyncio.gather(
            *[
                process_single_chunk(chunk_key=chunk_key, 
                                    chunk_content=chunk_dict["content"], 
                                    llm_client=llm_client, 
                                    prompt=entity_extract_prompt, 
                                    max_gleanning_time=max_gleanning_time) 
                for chunk_key, chunk_dict in chunks.items()
            ]
        )
    except Exception as e:
        import traceback
        print(f"处理过程中出现异常: {e}")
        print(traceback.format_exc())
        return None
    
    if len(results) < 1:
        logger.info("No new data !")
        return None
    
    nodes, edges = merge_nodes_and_edges(extracted_result=results)
    if len(nodes) < 1 and len(edges) < 1:
        logger.info("No new data !")
        return None
    
    # 合并字段信息
    updated_nodes, updated_edges = await merge_field_information_for_nodes_and_edges(
        nodes_and_edges=(nodes, edges), 
        config=config
    )
    
    # 插入节点和边到图中 
    await update_nodes_to_graph(updated_nodes, graph_instance)
    await update_edges_to_graph(updated_edges, graph_instance)
    
    # 更新实体到向量数据库
    if entity_vector_database is not None:
        data_for_vdb = {
            compute_id_by_mdhash(node_name, prefix="ent-"): {
                "content": node_name + " " + node_data.get("description", ""),
                "entity_name": node_name,
            }
            for node_name, node_data in updated_nodes.items()
            if node_name is not None and node_name != ""
        }
        await entity_vector_database.update_or_insert(data_for_vdb)

    return graph_instance


def get_context_base():
    return dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], # "<|>"
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"], # "##"
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"], # "<|COMPLETE|>"
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]), # ["organization", "person", "geo", "event"]
    )
    
async def gleanning(max_gleanning_time: int, llm_client, continue_prompt, history, if_loop_prompt, answer_last_time) -> str:
    final_answer = answer_last_time
    for i in range (max_gleanning_time):
        gleaned_response = await llm_client.chat(continue_prompt, history)
        
        if isinstance(gleaned_response, dict) and "choices" in gleaned_response:
            gleaned_answer = clean_think_tags(gleaned_response["choices"][0]["text"])
        else:
            logger.error(f"Gleaning response format unexpected: {type(gleaned_response)}")
            break
        
        history.extend([
            {"role": "user", "content": continue_prompt},
            {"role": "assistant", "content": gleaned_answer}
        ])
        
        final_answer += gleaned_answer
        if i == max_gleanning_time - 1:
            break
        
        # 还要多问一次llm要不要继续下去，我们可以直接合并在上面问
        
        if_loop_response = await llm_client.chat(if_loop_prompt, history)
        # 提取是否继续的答案
        if isinstance(if_loop_response, dict) and "choices" in if_loop_response:
            if_loop_result = if_loop_response["choices"][0]["text"]
        else:
            logger.error(f"If loop response format unexpected: {type(if_loop_response)}")
            break
            
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result != 'yes':
            break
    return final_answer

async def organize_entity_type_data_as_dict(record_attributes: list, chunk_key: str) :
    # judge if entity or not
    if len(record_attributes) != 4:
        logger.warning(f"Invalid entity record length: {len(record_attributes)}, expected 4")
        return False
    # judge if entity is valid
    entity_name = clean_str(record_attributes[1].upper())
    if entity_name is None or entity_name == "":
        return False
    
    entity_type = record_attributes[2]
    entity_description = record_attributes[3]
    entity_source_chunk_key = chunk_key
    
    return dict(
        entity_name = entity_name,
        entity_type = entity_type,
        entity_description = entity_description,
        entity_source_chunk = entity_source_chunk_key
    )

async def organize_relation_type_data_as_dict(record_attributes: list, chunk_key: str):
    # judge if relation or not
    if len(record_attributes) < 5:
        return False
    
    source_node = clean_str(record_attributes[1].upper())
    target_node = clean_str(record_attributes[2].upper())
    if source_node is None or target_node is None:
        logger.warning(f"Invalid node names: {record_attributes[1]}, {record_attributes[2]}")
        return False
    
    edge_description = clean_str(record_attributes[3].upper())
    edge_source_chunk = chunk_key
    weight = float(record_attributes[4]) if is_float_regex(record_attributes[-1]) else 1.0
    
    return dict(
        # name = tuple(soruce_node, target_node),
        source_node_id = source_node,
        target_node_id = target_node,
        edge_description = edge_description,
        edge_source_chunk = edge_source_chunk,
        weight = weight
    )
        

# return [nodes_dict, edges_dict]
async def process_single_chunk(chunk_key: str, chunk_content: str, 
                               llm_client, prompt, max_gleanning_time: int) -> dict[list[dict]] | None:
    # print(f"开始处理 chunk: {chunk_key[:10]}...")
    context_base = get_context_base()
    hint_prompt = prompt.format(**context_base, input_text=chunk_content)
    
    answer = await get_answer_from_qwen(llm_client, hint_prompt)
    
    # llm_response = await llm_client.chat(hint_prompt)
    # if isinstance(llm_response, dict) and "choices" in llm_response:
    #     answer = clean_think_tags(llm_response["choices"][0]["text"])
    # else:
    #     logger.info(f"LLM reposnse is unexpected: {type(llm_response)}")
    #     return
    
    history = pack_user_ass_to_openai_messages(hint_prompt, answer, using_amazon_bedrock=False)
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
        
    final_answer = await gleanning(max_gleanning_time=max_gleanning_time, llm_client=llm_client,
                             continue_prompt=continue_prompt, history=history,
                             if_loop_prompt=if_loop_prompt, answer_last_time=answer)
    
    # now we get a lot of records
    records = split_by_multimarkers(final_answer, context_base["record_delimiter"], context_base["completion_delimiter"])
    marker = context_base["tuple_delimiter"]
    record_attributes_list = clean_records(records=records, marker=marker)
    
    extracted_result = defaultdict(list)
    # 后续要把同名内容放在一起
    extracted_result["entity"] = []
    extracted_result["relationship"] = []
    
    # 可以先判断，然后避免很多不必要的api调用
    for record_attributes in record_attributes_list:
        
        if len(record_attributes) == 0:
            continue
        
        record_type = record_attributes[0]
        if record_type == "entity":
            func = organize_entity_type_data_as_dict
        elif record_type == "relationship":
            func = organize_relation_type_data_as_dict
        else:
            logger.info("Valid Graph Element Type During Extracting!!!")
            continue
        
        # 因为两个函数传入参数形式相同，可以这么做
        organized_result = await func(record_attributes=record_attributes, chunk_key=chunk_key)
        if isinstance(organized_result, dict):
            # a list of dict
            extracted_result[record_type].append(organized_result)
        else:
            logger.info("Not Valid Edge or Node !")
            continue
    
    return extracted_result

# combine data whose name is the same
def merge_nodes_and_edges(extracted_result: list[dict[list[dict]]]) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    # nodes = {node0: [data0,data1,data2], node1: [data0, data1,...]}
    nodes_after_merger = defaultdict(list)
    edges_after_merger = defaultdict(list)

    for res in extracted_result:
        
        if not res or not isinstance(res, dict):
            continue
            
        entity_list = res.get("entity", [])
        relation_list = res.get("relationship", [])

        # collect same nodes
        for ent in entity_list:
            entity_name = ent.get("entity_name")
            if entity_name and entity_name.strip() and isinstance(ent, dict):
                nodes_after_merger[entity_name].append(ent)

        # collect same edges
        for rel in relation_list:
            src = rel.get("source_node_id")
            tgt = rel.get("target_node_id")
            
            if (src and tgt and 
                src.strip() and tgt.strip() and 
                isinstance(rel, dict)):
                relation_name = tuple(sorted([src, tgt]))
                edges_after_merger[relation_name].append(rel)  
                    
    return nodes_after_merger, edges_after_merger


async def get_description_summary_from_llm(description: str, config: dict, llm_client, prompt) -> str | None:
        # 校验长度并处理（如果需要），如果小于summary长度，直接返回就可以，不需要调用LLM
        tokenizer = config["embedding_func"]
        summary_tokens_max_length = config["summary_tokens_max_length"]
        
        encoded_description = tokenizer.encode(description)
        if len(encoded_description) <= summary_tokens_max_length:
            return description
        else:
            llm_window_size = config["llm_window_size"]
            # description = tokenizer.decode(encoded_description)， 直接用原来的，不需要再解码一遍
            used_description = truncate_list(description, llm_window_size)
        
            context_base = get_context_base()
            hint_prompt = prompt.format(**context_base, input_text=used_description)
            
            answer = await get_answer_from_qwen(llm_client, hint_prompt)
            
        return answer
        
        


async def merge_field_information_for_nodes_and_edges(
    nodes_and_edges: tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]], 
    config: dict
) -> tuple[dict[str, MergedNodeFormat], dict[tuple[str, str], MergedEdgeFormat]]:
    """
    将相同实体和关系的多个记录合并为单个记录，进行字段累积合并
    """
    
    nodes_dict, edges_dict = nodes_and_edges
    
    nodes_to_be_updated = {}
    edges_to_be_updated = {}
    
    # 处理 nodes - 累积合并字段并去重，添加空值过滤
    for node_name, information_list in nodes_dict.items():
        # 新增：过滤空的或无效的信息
        valid_information_list = []
        for info in information_list:
            if (info and 
                isinstance(info, dict) and 
                info.get("entity_name") and 
                info.get("entity_name").strip()):
                valid_information_list.append(info)
        
        # 新增：如果没有有效信息，跳过这个节点
        if not valid_information_list:
            logger.warning(f"Skipping empty node: {node_name}")
            continue
        
        # 关键累积所有字段，而不是每次重新初始化
        all_types = []
        all_descriptions = []
        all_source_chunks = []
        
        # 收集这个节点的所有信息
        for information in valid_information_list:
            # 收集 entity_type（去重添加）
            entity_type = information.get("entity_type", "").strip()
            if entity_type and entity_type not in all_types:
                all_types.append(entity_type)  
            
            # 收集 entity_description（去重添加，保险起见）
            entity_description = information.get("entity_description", "").strip()
            if entity_description and entity_description not in all_descriptions:
                all_descriptions.append(entity_description)  
            
            # 收集 source_chunk_ids（去重添加）
            source_chunk = information.get("entity_source_chunk", "").strip()
            if source_chunk and source_chunk not in all_source_chunks:
                all_source_chunks.append(source_chunk)  
        
        # 新增：验证收集到的数据是否有效
        if not all_types and not all_descriptions and not all_source_chunks:
            logger.warning(f"No valid data collected for node: {node_name}")
            continue
        
        # 处理 description：如果描述太多，需要总结
        if len(all_descriptions) > 1:
            description_with_delimiter = GRAPH_FIELD_SEP.join(all_descriptions)
            final_description = await get_description_summary_from_llm(
                description=description_with_delimiter,
                config=config,
                llm_client=config["best_model_func"],
                prompt=PROMPTS["summarize_entity_descriptions"]
            )
            final_description = final_description or description_with_delimiter
        else:
            final_description = all_descriptions[0] if all_descriptions else ""
        
        # 新增：确保最终描述不为空
        if not final_description.strip():
            logger.warning(f"Empty description for node: {node_name}, using default")
            final_description = f"Entity: {node_name}"
        
        # 构建最终的合并节点 - 使用 GRAPH_FIELD_SEP 连接
        merged_node = {
            "type": GRAPH_FIELD_SEP.join(all_types) if all_types else "unknown",
            "description": final_description,
            "source_chunk_ids": GRAPH_FIELD_SEP.join(all_source_chunks) if all_source_chunks else ""
        }
        
        nodes_to_be_updated[node_name] = merged_node
    
    # 处理 edges -
    for edge_name, information_list in edges_dict.items():
        # 新增：过滤空的或无效的关系信息
        valid_information_list = []
        for info in information_list:
            if (info and 
                isinstance(info, dict) and 
                info.get("source_node_id") and 
                info.get("target_node_id") and
                info.get("source_node_id").strip() and
                info.get("target_node_id").strip()):
                valid_information_list.append(info)
        
        # 新增：如果没有有效信息，跳过这条边
        if not valid_information_list:
            logger.warning(f"Skipping empty edge: {edge_name}")
            continue
        
        source_id, target_id = edge_name
        
        # 累积所有边的字段
        all_weights = []
        all_descriptions = []
        all_source_chunks = []
        all_orders = []
        
        # 收集这条边的所有信息
        for information in valid_information_list:
            # 收集 weight（不去重，需要累加）
            weight = information.get("weight", 1.0)
            if isinstance(weight, (int, float)) and weight > 0:
                all_weights.append(weight)
            
            # 收集 edge_description（去重添加，保险起见）
            edge_description = information.get("edge_description", "").strip()
            if edge_description and edge_description not in all_descriptions:
                all_descriptions.append(edge_description)  
            
            # 收集 source_chunk_ids（去重添加）
            source_chunk = information.get("edge_source_chunk", "").strip()
            if source_chunk and source_chunk not in all_source_chunks:
                all_source_chunks.append(source_chunk)  
            
            # 收集 order（不去重，后续取最小值）
            order = information.get("order", 1)
            if isinstance(order, int) and order > 0:
                all_orders.append(order)  
        
        # 新增：验证收集到的数据是否有效
        if not all_descriptions and not all_source_chunks:
            logger.warning(f"No valid data collected for edge: {edge_name}")
            continue
        
        # 处理 weight：计算总和
        total_weight = sum(all_weights) if all_weights else 1.0
        
        # 处理 description：如果描述太多，需要总结
        if len(all_descriptions) > 1:
            description_with_delimiter = GRAPH_FIELD_SEP.join(all_descriptions)
            final_description = await get_description_summary_from_llm(
                description=description_with_delimiter,
                config=config,
                llm_client=config["best_model_func"],
                prompt=PROMPTS["summarize_entity_descriptions"]
            )
            final_description = final_description or description_with_delimiter
        else:
            final_description = all_descriptions[0] if all_descriptions else ""
        
        # 新增：确保最终描述不为空
        if not final_description.strip():
            final_description = f"Relationship between {source_id} and {target_id}"
        
        # 构建最终的合并边数据
        merged_edge_data = {
            "weight": total_weight,
            "description": final_description,
            "source_chunk_ids": GRAPH_FIELD_SEP.join(all_source_chunks) if all_source_chunks else "",
            "order": min(all_orders) if all_orders else 1
        }
        
        edges_to_be_updated[edge_name] = merged_edge_data
    
    return nodes_to_be_updated, edges_to_be_updated


async def update_nodes_to_graph(nodes: dict[str, MergedNodeFormat], graph_instance: BaseGraphStorage):
    """
    更新节点到图数据库中，支持多次插入，如果节点已存在则合并字段信息
    """
    for node_name, node_data_dict in nodes.items():
        # 添加节点名称验证
        if not node_name or not node_name.strip():
            logger.warning(f"Skipping invalid node name: {repr(node_name)}")
            continue
            
        try:
            node_already_in_graph = await graph_instance.get_node_data_by_id(node_name)
        except Exception as e:
            logger.error(f"Error getting node {node_name}: {e}")
            continue
        
        if node_already_in_graph is None:
            # 节点不存在，直接插入
            try:
                await graph_instance.update_or_insert_node(
                    node_id=node_name, 
                    node_data_part=node_data_dict
                )
                logger.info(f"Inserted new node: {node_name}")
            except Exception as e:
                logger.error(f"Error inserting node {node_name}: {e}")
                continue
        else:
            # 节点已存在，需要合并字段信息
            logger.info(f"Merging existing node: {node_name}")
            
            # 安全地获取现有数据
            existing_type = ""
            existing_description = ""
            existing_source_chunks = ""
            
            if isinstance(node_already_in_graph, dict):
                existing_type = node_already_in_graph.get("type", "")
                existing_description = node_already_in_graph.get("description", "")
                existing_source_chunks = node_already_in_graph.get("source_chunk_ids", "")
            else:
                # 处理 NetworkX AtlasView 等特殊对象
                try:
                    existing_type = getattr(node_already_in_graph, 'type', '') or node_already_in_graph.get('type', '') if hasattr(node_already_in_graph, 'get') else ""
                    existing_description = getattr(node_already_in_graph, 'description', '') or node_already_in_graph.get('description', '') if hasattr(node_already_in_graph, 'get') else ""
                    existing_source_chunks = getattr(node_already_in_graph, 'source_chunk_ids', '') or node_already_in_graph.get('source_chunk_ids', '') if hasattr(node_already_in_graph, 'get') else ""
                except:
                    logger.warning(f"Could not extract existing data for node {node_name}, using empty values")
            
            # 使用通用函数合并各个字段
            merged_type = merge_string_fields_with_separator(
                existing_value=str(existing_type),
                new_value=str(node_data_dict.get("type", ""))
            )
            
            merged_description = merge_string_fields_with_separator(
                existing_value=str(existing_description),
                new_value=str(node_data_dict.get("description", ""))
            )
            
            merged_source_chunks = merge_string_fields_with_separator(
                existing_value=str(existing_source_chunks),
                new_value=str(node_data_dict.get("source_chunk_ids", ""))
            )
            
            # 构建更新后的节点数据
            updated_node_data = {
                "type": merged_type,
                "description": merged_description,
                "source_chunk_ids": merged_source_chunks
            }
            
            # 更新节点
            try:
                await graph_instance.update_or_insert_node(
                    node_id=node_name, 
                    node_data_part=updated_node_data
                )
                # logger.info(f"Updated existing node: {node_name}")
            except Exception as e:
                logger.error(f"Error updating node {node_name}: {e}")
                continue
    

async def update_edges_to_graph(edges: dict[tuple[str, str], MergedEdgeFormat], graph_instance: BaseGraphStorage):
    """
    更新边到图数据库中，支持多次插入，如果边已存在则合并字段信息
    """
    for edge_name, edge_data_dict in edges.items():
        source_id, target_id = edge_name
        
        # 添加边验证
        if not source_id or not target_id or not source_id.strip() or not target_id.strip():
            # logger.warning(f"Skipping invalid edge: {repr(edge_name)}")
            continue
        
        # 新增：确保源节点和目标节点都存在
        try:
            source_exists = await graph_instance.check_node(source_id)
            target_exists = await graph_instance.check_node(target_id)
        except Exception as e:
            logger.error(f"Error checking nodes for edge {edge_name}: {e}")
            continue
        
        if not source_exists:
            # logger.warning(f"Source node '{source_id}' does not exist, skipping edge ({source_id}, {target_id})")
            continue
            
        if not target_exists:
            # logger.warning(f"Target node '{target_id}' does not exist, skipping edge ({source_id}, {target_id})")
            continue
        
        try:
            edge_already_in_graph = await graph_instance.get_edge_data_by_id(source_id, target_id)
        except Exception as e:
            logger.error(f"Error getting edge {edge_name}: {e}")
            continue
        
        if edge_already_in_graph is None:
            # 边不存在，直接插入
            try:
                await graph_instance.update_or_insert_edge(
                    edge_source_id=source_id,
                    edge_target_id=target_id,
                    edge_data_part=edge_data_dict
                )
                # logger.info(f"Inserted new edge: {source_id} -> {target_id}")
            except Exception as e:
                logger.error(f"Error inserting edge {edge_name}: {e}")
                continue
        else:
            # 边已存在，需要合并字段信息
            logger.info(f"Merging existing edge: {source_id} -> {target_id}")
            
            # 安全地获取现有数据
            existing_weight = 0.0
            existing_description = ""
            existing_source_chunks = ""
            existing_order = 1
            
            if isinstance(edge_already_in_graph, dict):
                existing_weight = float(edge_already_in_graph.get("weight", 0))
                existing_description = edge_already_in_graph.get("description", "")
                existing_source_chunks = edge_already_in_graph.get("source_chunk_ids", "")
                existing_order = edge_already_in_graph.get("order", 1)
            else:
                # 处理其他类型的数据
                try:
                    existing_weight = float(getattr(edge_already_in_graph, 'weight', 0) or edge_already_in_graph.get('weight', 0) if hasattr(edge_already_in_graph, 'get') else 0)
                    existing_description = str(getattr(edge_already_in_graph, 'description', '') or edge_already_in_graph.get('description', '') if hasattr(edge_already_in_graph, 'get') else "")
                    existing_source_chunks = str(getattr(edge_already_in_graph, 'source_chunk_ids', '') or edge_already_in_graph.get('source_chunk_ids', '') if hasattr(edge_already_in_graph, 'get') else "")
                    existing_order = int(getattr(edge_already_in_graph, 'order', 1) or edge_already_in_graph.get('order', 1) if hasattr(edge_already_in_graph, 'get') else 1)
                except:
                    logger.warning(f"Could not extract existing data for edge {edge_name}, using default values")
            
            merged_weight = existing_weight + float(edge_data_dict.get("weight", 0))
            
            merged_description = merge_string_fields_with_separator(
                existing_value=str(existing_description),
                new_value=str(edge_data_dict.get("description", ""))
            )
            
            merged_source_chunks = merge_string_fields_with_separator(
                existing_value=str(existing_source_chunks),
                new_value=str(edge_data_dict.get("source_chunk_ids", ""))
            )
            
            # 构建更新后的边数据
            updated_edge_data = {
                "weight": merged_weight,
                "description": merged_description,
                "source_chunk_ids": merged_source_chunks,
                "order": min(existing_order, edge_data_dict.get("order", 1))
            }
            
            # 更新边
            try:
                await graph_instance.update_or_insert_edge(
                    edge_source_id=source_id,
                    edge_target_id=target_id,
                    edge_data_part=updated_edge_data
                )
                # logger.info(f"Updated existing edge: {source_id} -> {target_id}")
            except Exception as e:
                logger.error(f"Error updating edge {edge_name}: {e}")
                continue
                



