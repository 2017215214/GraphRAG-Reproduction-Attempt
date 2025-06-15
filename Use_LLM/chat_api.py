from typing import Any, Dict, List, Optional, Union
import asyncio
import httpx
from Use_LLM.base_client import BaseQwenClient

class QwenChatAPI(BaseQwenClient):
    """专责 Chat Completion 调用，兼容 GraphRAG 格式"""
    
    def __init__(
        self,
        model: str,  # 必须指定模型名称
        api_url: str = "http://localhost:8001",
        timeout: float = 60.0,
        max_concurrency: int = 5,
        **kwargs
    ):
        super().__init__(
            api_url=api_url,
            timeout=timeout,
            max_concurrency=max_concurrency,
            **kwargs
        )
        self.model_name = model
    
    async def chat(
        self,
        prompt: str,
        history: Optional[List[Union[Dict, tuple]]] = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        retry_on_rate_limit: bool = True,
    ) -> Dict[str, Any]:
        # 构建消息历史
        msgs = self._build_messages(prompt, history)
        
        payload = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            data = await self._request(
                "POST", 
                "/v1/chat/completions", 
                json_data=payload
            )
            return self._format_response(data)
            
        except httpx.HTTPStatusError as e:
            if retry_on_rate_limit and e.response.status_code == 429:
                await asyncio.sleep(2)  # 等待后重试
                return await self.chat(prompt, history, temperature, max_tokens, False)
            raise
    
    def _build_messages(self, prompt: str, history) -> List[Dict]:
        """构建符合OpenAI格式的消息列表"""
        msgs = []
        
        # 处理历史记录
        for item in history or []:
            if isinstance(item, dict) and "role" in item:
                msgs.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                msgs.append({"role": "user", "content": item[0]})
                if item[1]:  # 非空回复
                    msgs.append({"role": "assistant", "content": item[1]})
            else:
                raise ValueError(f"无效的历史记录格式: {type(item)}")
        
        # 添加当前提示
        msgs.append({"role": "user", "content": prompt})
        return msgs
    
    def _format_response(self, data: Dict) -> Dict[str, Any]:
        """标准化响应格式"""
        choice = data["choices"][0]
        message = choice["message"]
        
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": data.get("created", 0),
            "model": data.get("model", self.model_name),
            "choices": [{
                "index": choice.get("index", 0),
                "text": message["content"],
                "role": message["role"],
                "finish_reason": choice.get("finish_reason", "stop")
            }],
            "usage": data.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }