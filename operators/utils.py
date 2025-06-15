from dataclasses import dataclass
from functools import wraps
import numpy as np
import os
import json
import logging
import asyncio
from hashlib import md5
import re
import html
from dataclasses import asdict, is_dataclass
from dataclasses import is_dataclass
from typing import Any, Dict
import inspect
import numbers
from transformers import AutoTokenizer


@dataclass
class EmbeddingFunction:
    
    max_token_size: int
    embedding_dim: int
    func: callable
    
    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)
    
    
# 从硬盘加载
def load_json_file(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)
    
logger = logging.getLogger("dyh")

# 写回硬盘
def write_json(json_object, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_object, f, indent=2, ensure_ascii=False)
        
        
# 获取事件循环
def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        logger.info("Creating a new event loop.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=loop)
    return loop
        
# 用hash编码数据库单个元素的名称
def compute_id_by_mdhash(content: str, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(func, max_size:int, waitting_time: float = 0.0001):
    _current_size = 0
    
    # 利用被装饰的函数进行操作，得到result，所以内层函数return结果（和原函数最接近）
    # 外层的，返回这个新函数就好
    async def restricted_func(*args, **kwargs):
        nonlocal _current_size
        while _current_size >= max_size:
            await asyncio.sleep(waitting_time)
        # if current_thread num < max_thread num
        _current_size += 1
        result = await func(*args, **kwargs)
        _current_size -= 1
        return result
        
    return restricted_func

# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(prompt: str, generated_content: str, using_amazon_bedrock: bool):
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_content},
        ]
        
def split_by_multimarkers(text: str, record_delimiter: str, completion_delimiter: str) -> list[str]:
    """
    根据记录分隔符和完成分隔符将文本分割成记录列表，增强容错性
    """
    # 首先按完成分隔符分割，取第一部分
    parts = text.split(completion_delimiter)
    main_content = parts[0] if len(parts) > 1 else text
    
    # 按记录分隔符分割
    records = main_content.split(record_delimiter)
    
    # 清理并修复不完整的记录
    cleaned_records = []
    for i, record in enumerate(records):
        record = record.strip()
        if not record:
            continue
            
        # 检查记录是否包含不完整的实体定义
        # 如果记录中间出现 ("entity" 或 ("relationship"，说明格式有问题
        if '("entity"' in record[10:] or '("relationship"' in record[10:]:  # 排除开头的正常情况
            # 尝试修复：在第一个 ( 前截断
            if record.startswith('("'):
                # 找到第一个不完整实体的位置
                next_entity_pos = max(
                    record.find('("entity"', 10),
                    record.find('("relationship"', 10)
                )
                if next_entity_pos > 0:
                    # 截断到不完整实体之前，并添加结束符
                    truncated = record[:next_entity_pos].rstrip()
                    if not truncated.endswith('")'):
                        truncated += '")'
                    cleaned_records.append(truncated)
                    
                    # 处理剩余部分作为新记录
                    remaining = record[next_entity_pos:]
                    if remaining.strip():
                        records.insert(i + 1, remaining)
                    continue
        
        cleaned_records.append(record)
    
    return cleaned_records


def clean_str(input) -> str:
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    # 该正则表达式匹配 ASCII 范围内的控制字符，具体范围：
    # \x00-\x1f：匹配从 0 到 31 的控制字符（例如换行符、回车符、制表符等）。
    # \x7f-\x9f：匹配 127 到 159 之间的控制字符。
    
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def get_serializable_config(config: Any) -> Dict:
    """
    只过滤掉真正不可序列化的对象，保留类型和函数引用。
    
    Args:
        config: BaseConfig 实例
    
    Returns:
        dict: 可序列化但保留必要引用的配置字典
    """
    def is_unserializable(obj: Any) -> bool:
        """判断对象是否不可序列化但需要保留"""
        # 保留类型引用
        if isinstance(obj, type):
            return False
        # 保留函数引用    
        if callable(obj):
            return False
        
        # 特别处理：保留 QwenChatAPI 和 SentenceTransformer 等重要对象
        from Use_LLM.chat_api import QwenChatAPI
        from sentence_transformers import SentenceTransformer
        
        if isinstance(obj, (QwenChatAPI, SentenceTransformer)):
            return False  # 不过滤这些重要对象
            
        # 检查是否含有 __dict__ 但不可序列化
        if hasattr(obj, '__dict__'):
            try:
                from json import dumps
                dumps(obj.__dict__)
                return False
            except:
                return True
        return False

    def safe_asdict(obj: Any) -> Dict:
        if not is_dataclass(obj):
            return obj
        
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            
            # 跳过真正不可序列化的对象
            if is_unserializable(value):
                continue
                
            try:
                # 递归处理嵌套的数据类
                result[field_name] = safe_asdict(value)
            except Exception:
                # 如果处理失败，则保留原始值（这很重要！）
                result[field_name] = value
                
        return result

    return safe_asdict(config)

def clean_think_tags(text: str) -> str:
    """移除 <think> 标签及其内容"""
    import re
    # 移除 <think>...</think> 内容
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def clean_records(records: list[str], marker: str) -> list[list[str]]:
    """
    从记录列表中提取属性列表
    
    Args:
        records: 记录列表，每个记录都是括号包围的字符串
        marker: 字段分隔符，如 "<|>"
    
    Returns:
        list[list[str]]: 每个记录的属性列表
    """
    record_attributes_list = []
    
    for record in records:
        # 使用正则表达式提取括号内的内容
        match = re.search(r'\((.*)\)', record)
        if match is None:
            continue
        
        # 获取括号内的内容
        inner_content = match.group(1)
        
        # 按分隔符分割成属性列表
        attributes = inner_content.split(marker)
        
        # 清理每个属性的首尾空格和引号
        cleaned_attributes = []
        for attr in attributes:
            # 移除首尾空格和可能的引号
            cleaned_attr = attr.strip().strip('"').strip("'")
            cleaned_attributes.append(cleaned_attr)
        
        record_attributes_list.append(cleaned_attributes)
    
    return record_attributes_list


def extend_elements_to_list_if_not_exists(elements: list, lst: list) -> None:

    for element in elements:
        """类似update, 如果有就不加了"""
        if element not in lst:
            list.extend(element)
            
def truncate_list(list_data: list, max_length: int) -> list:
    """
        截断到指定长度
    """
    if len(list_data) > max_length:
        return list_data[:max_length]
    # Python 列表切片超出长度不会报错
    return list_data


def merge_string_fields_with_separator(existing_value: str, new_value: str, separator: str = "<SEP>") -> str:
    """
    合并两个用分隔符连接的字符串字段，去重并排序
    
    Args:
        existing_value: 已存在的字段值
        new_value: 新的字段值
        separator: 分隔符，默认为 GRAPH_FIELD_SEP
    
    Returns:
        str: 合并后的字符串
    """
    # 分割现有值
    existing_list = existing_value.split(separator) if existing_value else []
    # 分割新值
    new_list = new_value.split(separator) if new_value else []
    
    # 合并并去重
    all_items = list(set(existing_list + new_list))
    # 过滤空字符串
    all_items = [item for item in all_items if item.strip()]
    
    # 排序并用分隔符连接
    return separator.join(sorted(all_items))


def visualize_community_schema(schema, max_nodes_display=5, max_edges_display=5, max_desc_chars=30):
    """
    可视化社区结构，以层次化、美观的方式展示
    
    参数:
        schema: 社区schema字典
        max_nodes_display: 每个社区最多显示的节点数
        max_edges_display: 每个社区最多显示的边数
        max_desc_chars: 描述信息最多显示的字符数
    """
    if not schema:
        print("社区结构为空")
        return
    
    # 按层级和社区ID排序
    communities_by_level = {}
    for cid, data in schema.items():
        level = data.get('level', 0)
        if level not in communities_by_level:
            communities_by_level[level] = []
        communities_by_level[level].append((cid, data))
    
    # 打印层次化的社区结构
    print("## 社区结构概览")
    print(f"总社区数: {len(schema)}\n")
    
    # 按层级打印社区
    for level in sorted(communities_by_level.keys()):
        print(f"### 层级 {level} (共 {len(communities_by_level[level])} 个社区)")
        
        # 按社区ID排序
        sorted_communities = sorted(communities_by_level[level], key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        
        for cid, data in sorted_communities:
            # 社区基本信息
            print(f"\n#### 社区 {cid}: {data['title']}")
            
            # 获取描述信息 (尝试多个可能的字段名)
            description = None
            for desc_field in ['description', 'report_string', 'report', 'summary']:
                if desc_field in data and data[desc_field]:
                    description = data[desc_field]
                    break
                    
            # 显示描述信息
            if description:
                short_desc = description[:max_desc_chars] + "..." if len(description) > max_desc_chars else description
                print(f"- 描述: {short_desc}")
            
            # 节点信息
            nodes = data.get('nodes', [])
            print(f"- 节点数量: {len(nodes)}")
            if nodes:
                print(f"- 代表节点: {', '.join(nodes[:max_nodes_display])}" + 
                      (f" ... 等 {len(nodes)} 个" if len(nodes) > max_nodes_display else ""))
            
            # 边信息
            edges = data.get('edges', [])
            print(f"- 边数量: {len(edges)}")
            if edges and max_edges_display > 0:
                print("- 代表边:")
                for i, edge in enumerate(edges[:max_edges_display]):
                    print(f"  - {edge[0]} ←→ {edge[1]}")
                if len(edges) > max_edges_display:
                    print(f"  - ... 等 {len(edges)} 条边")
            
            # 子社区信息
            sub_communities = data.get('sub_communities', [])
            if sub_communities:
                print(f"- 子社区: {', '.join(sub_communities)}")
            
            # 其他统计信息
            print(f"- 重要性指数: {data.get('occurrence', 0):.2f}")
            
            print("-" * 40)
        
        print("\n" + "=" * 80 + "\n")

    
def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)
    
    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")
    
    return prediction_json

def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}
    
    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
    
    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    
    return extracted_values

def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None

def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist
        
# 你返回一个list，原始数据不懂，也不单独提取出来某几个字段，单纯就是截取片段出来
def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string(key(data), model_name="Qwen/Qwen3-8B"))
        if tokens > max_token_size:
            return list_data[:i]
    # print(f"tokens accumulated: {tokens}")
    return list_data

def encode_string(content: str, model_name: str) -> list[int]:
    """
    将一段文本编码成 token id 列表，如果加载 tokenizer 失败则返回空列表。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}")
        # 直接返回空列表，避免后续 UnboundLocalError
        return []

    # 调用 tokenizer 编码
    batch = tokenizer(content)
    # batch["input_ids"] 形如 (1, L)
    return batch["input_ids"]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )

def truncate_prompt(
    prompt: str,
    query: str,
    model_name: str,
    max_token_size: int
) -> str:
    """
    在一个窗口大小 max_token_size 内，构造形如：
      prefix (包含 Question: query)
      + truncated prompt
      + suffix (重复 query)
    保证 query 不被截断。
    """
    # 定义前缀和后缀
    prefix = (
        "Please answer the following question using the prompt below:\n\n"
        f"Question: {query}\n\n"
        "Prompt:\n"
    )
    # suffix = f"\n\n Answer concisely without restating the question."

    # 加载 tokenizer（trust_remote_code 依项目需求开启/关闭）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 计算前缀和后缀的 token 数
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    # suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    # remaining = max_token_size - len(prefix_ids) - len(suffix_ids)
    remaining = max_token_size - len(prefix_ids)

    # 如果没有预算，就不保留原 prompt
    if remaining <= 0:
        body = ""
    else:
        # 对原 prompt 按剩余预算截断
        truncated_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=remaining,
            add_special_tokens=False
        )["input_ids"]
        body = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    # 返回最终合并结果
    # return prefix + body + suffix
    return prefix + body

    
async def get_answer_from_qwen(client, prompt: str, history: str = "", query: str = "") -> str:
    system_prompt = """
You are a helpful assistant, but only output your final answer in the requested format. Do not include your thinking process in the response.
"""
    prompt = system_prompt + prompt
    if history not in [None, ""]:
        llm_response = await client.chat(prompt=prompt, history=history)
        
    else:
        llm_response = await client.chat(prompt=prompt)
        
    if isinstance(llm_response, dict) and "choices" in llm_response:
        answer = clean_think_tags(llm_response["choices"][0]["text"])
        return answer
    else:
        logger.info(f"LLM reposnse is unexpected: {type(llm_response)}")
        return