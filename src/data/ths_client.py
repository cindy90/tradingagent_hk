"""同花顺 iFinD QuantAPI 客户端（HTTP REST 实现）。

文档: https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/

认证流程:
  refresh_token (用户配置) → get_access_token → access_token (缓存到本地，TTL 内复用)
  调用 report_query / 其它接口时在 Header 里带 access_token。

当前实现的接口:
  - get_access_token / update_access_token  (token 管理)
  - report_query                            (公司公告查询)
  - download_pdf                            (按 pdfURL 下载文件)

便利方法:
  - find_prospectus(ticker)        从公告流里找招股书相关文件
  - auto_fetch_prospectus(ticker)  自动下载到 data/prospectus/<ticker>.pdf

未配置 refresh_token 时所有方法返回空 / None，调用方应有 fallback。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

from .cache import disk_cache

BASE_URL = "https://quantapi.51ifind.com/api/v1"

# 港股招股书相关公告标题的关键词（按优先级排序）。
# 若同花顺 reportType 没有专门的"招股书"分类，则用 title 关键词二次过滤。
PROSPECTUS_TITLE_KEYWORDS = [
    "聆讯后资料集",   # PHIP - Post Hearing Information Pack
    "招股章程",       # Prospectus (HK 用语)
    "招股书",
    "Prospectus",
    "申请版本",       # Application Proof
    "PHIP",
]


class THSAPIError(RuntimeError):
    pass


class THSClient:
    """同花顺 iFinD QuantAPI 客户端。"""

    def __init__(
        self,
        refresh_token: str | None = None,
        token_cache_path: str | Path | None = None,
        token_ttl: int | None = None,
    ):
        s = get_settings()
        self.refresh_token = refresh_token or s.ths_refresh_token
        self.token_ttl = token_ttl if token_ttl is not None else s.ths_token_ttl
        if token_cache_path:
            self._token_cache = Path(token_cache_path)
        elif s.ths_token_cache:
            self._token_cache = Path(s.ths_token_cache)
        else:
            self._token_cache = s.cache_dir / "ths_access_token.json"
        self._access_token: str | None = None
        self._access_token_ts: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.refresh_token)

    # ---------- token 管理 ----------

    def _load_cached_token(self) -> tuple[str | None, float]:
        if not self._token_cache.exists():
            return None, 0.0
        try:
            data = json.loads(self._token_cache.read_text(encoding="utf-8"))
            return data.get("access_token"), data.get("ts", 0.0)
        except Exception:
            return None, 0.0

    def _save_cached_token(self, token: str) -> None:
        self._token_cache.parent.mkdir(parents=True, exist_ok=True)
        self._token_cache.write_text(
            json.dumps({"access_token": token, "ts": time.time()}, ensure_ascii=False),
            encoding="utf-8",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _request_access_token(self, force_refresh: bool = False) -> str:
        if not self.refresh_token:
            raise THSAPIError("refresh_token 未配置，请在 .env 里设置 THS_REFRESH_TOKEN")
        url = f"{BASE_URL}/{'update_access_token' if force_refresh else 'get_access_token'}"
        headers = {
            "Content-Type": "application/json",
            "refresh_token": self.refresh_token,
        }
        with httpx.Client(timeout=30) as c:
            r = c.post(url, headers=headers)
            r.raise_for_status()
            payload = r.json()
        # 兼容两种返回结构：{data: {access_token: ...}} 或 {access_token: ...}
        token = (payload.get("data") or {}).get("access_token") or payload.get("access_token")
        if not token:
            raise THSAPIError(f"获取 access_token 失败: {payload}")
        return token

    def get_access_token(self, force_refresh: bool = False) -> str:
        if force_refresh:
            tok = self._request_access_token(force_refresh=True)
            self._access_token = tok
            self._access_token_ts = time.time()
            self._save_cached_token(tok)
            return tok

        if self._access_token and (time.time() - self._access_token_ts) < self.token_ttl:
            return self._access_token

        cached_tok, cached_ts = self._load_cached_token()
        if cached_tok and (time.time() - cached_ts) < self.token_ttl:
            self._access_token, self._access_token_ts = cached_tok, cached_ts
            return cached_tok

        tok = self._request_access_token(force_refresh=False)
        self._access_token = tok
        self._access_token_ts = time.time()
        self._save_cached_token(tok)
        return tok

    # ---------- 通用请求 ----------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _post(self, endpoint: str, body: dict, retry_on_auth: bool = True) -> dict:
        token = self.get_access_token()
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "access_token": token,
        }
        with httpx.Client(timeout=60) as c:
            r = c.post(url, headers=headers, json=body)
        if r.status_code == 401 and retry_on_auth:
            logger.warning("access_token 失效，强制刷新后重试")
            self.get_access_token(force_refresh=True)
            return self._post(endpoint, body, retry_on_auth=False)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and payload.get("errorcode") not in (None, 0, "0"):
            raise THSAPIError(f"{endpoint} 接口错误: {payload}")
        return payload

    # ---------- report_query ----------

    def query_reports(
        self,
        codes: list[str] | str,
        output_fields: list[str] | str | None = None,
        function_params: dict[str, Any] | None = None,
    ) -> list[dict]:
        """调用 /report_query 公告查询接口。

        参数:
            codes:            证券代码列表，港股形如 "00700.HK" 或 "09999.HK"
            output_fields:    返回字段，默认包含日期/时间/标题/PDF链接
            function_params:  公告类型 / 时间区间 / 关键词等过滤项

        返回: 字典列表，每条公告一个 dict。
        """
        if not self.configured:
            return []

        if isinstance(codes, list):
            codes_str = ",".join(codes)
        else:
            codes_str = codes

        if output_fields is None:
            output_fields = [
                "thscode", "secName", "declareDate", "publishTime",
                "title", "reportType", "pdfURL",
            ]
        if isinstance(output_fields, list):
            output_para = ",".join(output_fields)
        else:
            output_para = output_fields

        # functionpara 同花顺通常要求传 dict
        body = {
            "codes": codes_str,
            "outputpara": output_para,
            "functionpara": function_params or {},
        }
        try:
            payload = self._post("report_query", body)
        except Exception as e:
            logger.warning(f"report_query failed: {e}")
            return []

        # 不同 API 版本字段名不一致，做兼容
        rows = (
            payload.get("tables")
            or payload.get("data")
            or payload.get("result")
            or []
        )
        if isinstance(rows, dict):
            # 形如 {time:[...], title:[...]} 列式 → 转成行式
            keys = list(rows.keys())
            if keys and isinstance(rows[keys[0]], list):
                n = len(rows[keys[0]])
                return [{k: rows[k][i] for k in keys} for i in range(n)]
            return [rows]
        return rows if isinstance(rows, list) else []

    # ---------- 招股书便利方法 ----------

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def find_prospectus(
        self,
        ticker: str,
        start_date: str = "2023-01-01",
        end_date: str | None = None,
    ) -> list[dict]:
        """查找某 ticker 的招股书相关公告。

        ticker 接受 "09999" / "09999.HK" 两种形式，自动补全后缀。
        """
        code = ticker if "." in ticker else f"{ticker.zfill(5)}.HK"
        end_date = end_date or time.strftime("%Y-%m-%d")

        # 先按时间范围拉一批，再用标题关键词过滤
        rows = self.query_reports(
            codes=code,
            function_params={"startDate": start_date, "endDate": end_date},
        )

        def is_prospectus(row: dict) -> bool:
            title = str(row.get("title", "") or row.get("TITLE", ""))
            return any(kw.lower() in title.lower() for kw in PROSPECTUS_TITLE_KEYWORDS)

        candidates = [r for r in rows if is_prospectus(r)]
        # 按发布日期降序，最新的在前
        def _date_key(r: dict) -> str:
            return str(r.get("declareDate") or r.get("DECLAREDATE") or r.get("publishTime") or "")
        candidates.sort(key=_date_key, reverse=True)
        return candidates

    def download_pdf(self, url: str, dest: str | Path) -> bool:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=120, follow_redirects=True) as c:
                r = c.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
            logger.info(f"招股书下载完成: {dest} ({len(r.content)/1024:.0f} KB)")
            return True
        except Exception as e:
            logger.warning(f"download_pdf 失败 {url}: {e}")
            return False

    def auto_fetch_prospectus(
        self,
        ticker: str,
        dest_dir: str | Path | None = None,
        prefer: str = "PHIP",
    ) -> Path | None:
        """自动定位并下载招股书 PDF 到本地。

        prefer: 标题关键词偏好。"PHIP" 优先聆讯后资料集；"Application" 优先申请版。
        返回本地 PDF 路径或 None。
        """
        if not self.configured:
            logger.info("THS 未配置 refresh_token，跳过自动获取招股书")
            return None
        s = get_settings()
        dest_dir = Path(dest_dir) if dest_dir else s.prospectus_dir
        local_pdf = dest_dir / f"{ticker.split('.')[0]}.pdf"
        if local_pdf.exists():
            logger.info(f"招股书已在本地: {local_pdf}")
            return local_pdf

        candidates = self.find_prospectus(ticker)
        if not candidates:
            logger.warning(f"未找到 {ticker} 的招股书相关公告")
            return None

        # prefer 排序：把含 prefer 关键词的项前置
        candidates.sort(key=lambda r: prefer.lower() not in str(r.get("title", "")).lower())

        for row in candidates:
            url = row.get("pdfURL") or row.get("PDFURL") or row.get("pdf_url")
            if not url:
                continue
            if self.download_pdf(url, local_pdf):
                return local_pdf
        logger.warning(f"{ticker} 所有候选公告 PDF 下载均失败")
        return None
