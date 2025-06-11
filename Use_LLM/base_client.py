import asyncio
import httpx
import numpy as np
from typing import List, Optional, Dict, Any

class BaseQwenClient:
    """优化后的底层 HTTP 客户端"""
    def __init__(
        self,
        api_url: str = "http://localhost:8001",  # 默认本地地址
        timeout: float = 60.0,
        max_concurrency: int = 10,
        retry_times: int = 3,
        retry_backoff: float = 1.0
    ):
        # 确保地址以 / 结尾
        # self.api_url = api_url.rstrip("/") + "/v1"  # 添加 /v1 路径
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0)
        )
        self.api_url = api_url.rstrip("/")  # 确保没有多余的斜杠
        self._sem = asyncio.Semaphore(max_concurrency)
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff

    async def _request(self, method: str, endpoint: str, json_data: Dict) -> Dict:
        """增强的错误处理和重试逻辑"""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        async with self._sem:
            for attempt in range(self.retry_times + 1):
                try:
                    response = await self._client.request(
                        method, url, json=json_data
                    )
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, httpx.RequestError) as e:
                    if attempt < self.retry_times:
                        wait = self.retry_backoff * (2 ** attempt)
                        print(f"尝试 {attempt+1}/{self.retry_times} 失败，{wait}秒后重试: {str(e)}")
                        await asyncio.sleep(wait)
                    else:
                        raise RuntimeError(f"请求失败: {str(e)}") from e

    async def aclose(self):
        await self._client.aclose()