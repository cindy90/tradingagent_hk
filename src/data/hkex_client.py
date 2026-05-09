"""港交所披露易客户端（骨架）。

公开接口可通过 https://www1.hkexnews.hk 检索申请版/聆讯后资料集。
此处保留接口形态，待具体实现。
"""
from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from .cache import disk_cache

HKEX_NEWS_BASE = "https://www1.hkexnews.hk"


@disk_cache(ttl_seconds=24 * 3600, namespace="hkex")
def search_listing_documents(company_keyword: str) -> list[dict[str, Any]]:
    """检索某公司在港交所披露的上市相关文件（申请版招股书、聆讯后资料集等）。

    返回示例：
        [{"date": "2025-03-12", "title": "聆讯后资料集", "url": "https://..."}, ...]

    TODO: 解析 HKEX news 站内检索接口或公开 API。
    """
    logger.debug(f"search_listing_documents stub: {company_keyword}")
    return []


def download_document(url: str, dest_path: str) -> bool:
    """下载披露易上的 PDF 文件到本地。"""
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(r.content)
        return True
    except Exception as e:
        logger.warning(f"download_document failed: {e}")
        return False
