from langchain.tools import tool
import httpx
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


@tool
async def check_links(urls: List[str]) -> str:
    """验证URL列表的有效性，确保链接可以正常访问。

    Args:
        urls: 需要验证的URL列表

    Returns:
        JSON格式的验证结果，包含每个URL的状态
    """
    import json
    
    if not urls:
        return json.dumps({"valid": [], "invalid": [], "total": 0}, ensure_ascii=False)
    
    results = {"valid": [], "invalid": [], "total": len(urls)}
    
    async def check_single_url(url: str) -> tuple[str, bool, str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(url, follow_redirects=True)
                if response.status_code < 400:
                    return url, True, "OK"
                else:
                    return url, False, f"HTTP {response.status_code}"
        except Exception as e:
            return url, False, str(e)
    
    tasks = [check_single_url(url) for url in urls]
    check_results = await asyncio.gather(*tasks)
    
    for url, is_valid, message in check_results:
        if is_valid:
            results["valid"].append({"url": url, "status": message})
        else:
            results["invalid"].append({"url": url, "error": message})
    
    return json.dumps(results, ensure_ascii=False, indent=2)