from operators.utils import (
    compute_id_by_mdhash,
    logger,
)

from my_graphrag.base_config import BaseConfig
from transformers import AutoTokenizer
import numpy as np


def process_docs_to_new_fromat(docs: list[str]):
    # new_docs = {compute_id_by_hash(cleaned_docs, prefix="doc-"): {"conten": cleaned_doc}}
    new_format_docs = {}
    for doc in docs:
        cleaned_doc = doc.strip()
        prefix_id = compute_id_by_mdhash(content=cleaned_doc, prefix="doc-")
        new_format_docs[prefix_id] = {"content": cleaned_doc}
    return new_format_docs
        




# 想清楚chunk_func能处理list[int]，还是list[list[int]]，以此确定for循环写在哪里
def split_texts_into_chunks(texts: dict[str, dict], config: dict) -> dict:
    """Split texts into chunks using the specified model's tokenizer"""
    
    # 使用 Qwen 的原生 tokenizer 替代 tiktoken
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load {config['model_name']} tokenizer: {e}")
        raise

    # 修复拼写错误并处理文本
    texts_content: list[str] = []
    texts_keys: list[str] = []
    for k, v in texts.items():
        texts_content.append(v["content"])  # 修复拼写错误
        texts_keys.append(k)
    
    # 批量编码
    try:
        # transformers tokenizer 的批处理
        tokens_batch = tokenizer(
            texts_content,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        # 转换为 list[list[int]] 格式以保持接口兼容
        tokens = tokens_batch['input_ids'].tolist()
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        

    chunk_token_size: int = config["chunk_token_size"]
    overlap_token_size: int = config["overlap_token_size"]
    chunk_func = config["chunk_func"]
    # 调用分块函数
    inserting_chunks = chunk_func(
        tokens, 
        texts_keys, 
        tokenizer,
        chunk_token_size,
        overlap_token_size
    )
    return inserting_chunks
    
def chunking_by_token_size(
    texts_content: list[list[int]], 
    text_keys: list[str],
    tokenizer,
    chunk_token_size: int,
    overlap_token_size: int
) -> dict[str, dict]:
    
    chunks = {}
    for i, content in enumerate(texts_content):
        # 获取每个块的token
        chunk_tokens = [
            content[j: j+chunk_token_size] 
            for j in range(0, len(content), chunk_token_size-overlap_token_size)
        ]
        
        # 解码每个块的内容 - 修改这里
        chunk_contents = [
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in chunk_tokens
        ]
        original_doc_id = text_keys[i]
        
        # 为每个块创建entry
        for chunk_idx, (tokens, content) in enumerate(zip(chunk_tokens, chunk_contents)):
            chunk_key = compute_id_by_mdhash(content=content, prefix="chunk-")
            
            # 正确创建chunk字典，保存token长度而不是tokens本身
            chunk = {
                chunk_key: {
                    "tokens": len(tokens),  # 只保存长度
                    "content": content,
                    "original_doc_id": original_doc_id,
                    "chunk_order_index": chunk_idx
                }
            }
            chunks.update(chunk)
    
    return chunks