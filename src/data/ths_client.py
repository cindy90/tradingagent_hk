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

# THS_BD 常用基础信息字段（公司层面静态/低频数据）。
# 字段名按 iFinD 文档命名；不同套餐可能有差异，缺失字段会自动忽略。
DEFAULT_BASIC_DATA_INDICATORS_HK = [
    "ths_corp_chi_name_stock",           # 公司中文名称
    "ths_corp_eng_name_stock",           # 公司英文名称
    "ths_main_business_stock",           # 主营业务
    "ths_listed_date_stock",             # 上市日期
    "ths_thscode_industry_name_stock",   # 同花顺行业
    "ths_gics_lv4_industry_stock",       # GICS 四级行业
    "ths_actual_controller_stock",       # 实际控制人
    "ths_top_holders_stock",             # 主要股东
    "ths_register_capital_stock",        # 注册资本
    "ths_employee_num_stock",            # 员工人数
    "ths_business_scope_stock",          # 经营范围
    "ths_office_addr_stock",             # 办公地址
]

# 港股市场宏观与流动性相关 EDB 指标编码。
#
# ⚠️ 真实编码必须从 iFinD 客户端"超级命令"工具按指标名搜索得到（quantapi 文档无公开字典）。
# 旧版本这里写的 M00xxxx 全是占位符（M001620251 实测返回的是"GDP:第二产业:建筑业"），
# 会让 macro agent 误以为有数据，然后 LLM 凭训练知识幻觉出 2024 年数字。
#
# 真实编码格式是 L00xxxxxx（不是 M00xxxxxx），从 iFinD 超级命令工具粘贴的 THS_EDB('L001619084;...')
# 里提取分号前后的编码即可。
#
# 待补全（从 iFinD 客户端超级命令搜以下关键词）：
#   "HIBOR_3M":            搜 "HIBOR 3 个月" / "HIBOR3M"
#   "USD_HKD":             搜 "美元兑港币" / "USDHKD"
#   "HSI_PE":              搜 "恒生指数 市盈率 TTM"
#   "HK_IPO_AMOUNT":       搜 "香港 IPO 集资额" 或 "新股 集资"
#   "SOUTHBOUND_NET_FLOW": 搜 "港股通 累计净买入" / "南向资金"
#   "CN_PMI":              搜 "中国 制造业 PMI"
# 真实编码（已通过 iFinD QuantAPI 实测验证 index_name 与数据返回）
DEFAULT_HK_MACRO_EDB_CODES: dict[str, str] = {
    "HIBOR_1M":                 "L001619084",  # 1 个月港元 HIBOR
    "HIBOR_3M":                 "L001619085",  # 3 个月港元 HIBOR
    "HSI_LEVEL":                "G002856503",  # 恒生指数点位（日频）
    "HSI_PE_TTM":               "G020673147",  # 中国香港:市盈率:恒生指数
    "HK_IPO_RAISED_MONTHLY":    "G002701953",  # 募集资金:IPO:新发行股份:香港联交所（月频，亿港元）
    "SOUTHBOUND_NET_HKD_DAILY": "S006015793",  # 港股通:净买入额（HKD，日频）
    "SOUTHBOUND_CUM_CNY":       "S006015794",  # 港股通:累计净买入额（CNY）
    "USD_HKD_FWD_ON":           "L035711198",  # USD/HKD 外汇远期曲线 ON
    "USD_HKD_SWAP_1Y":          "M021799523",  # USD/HKD 外汇掉期曲线 1Y
    "CN_PMI":                   "M002043802",  # 中国制造业 PMI（月频）
}


class THSAPIError(RuntimeError):
    pass


def _parse_table_to_dict_by_code(
    payload: dict, codes: list[str], indicators: list[str]
) -> dict[str, dict]:
    """basic_data 返回格式兼容解析：把行式/列式 payload 转成 {code: {indicator: value}}."""
    rows = (
        payload.get("tables")
        or payload.get("data")
        or payload.get("result")
        or []
    )
    out: dict[str, dict] = {code: {} for code in codes}

    if isinstance(rows, dict):
        # 列式：{thscode: [...], indicator1: [...], indicator2: [...]}
        keys = list(rows.keys())
        if not keys:
            return out
        n = len(rows[keys[0]]) if isinstance(rows[keys[0]], list) else 0
        for i in range(n):
            code_key = "thscode" if "thscode" in rows else ("THSCODE" if "THSCODE" in rows else None)
            code = rows[code_key][i] if code_key else codes[i] if i < len(codes) else None
            if code is None:
                continue
            for k in keys:
                if k.lower() == "thscode":
                    continue
                out.setdefault(code, {})[k] = rows[k][i] if isinstance(rows[k], list) else None
        return out

    if isinstance(rows, list):
        for r in rows:
            code = r.get("thscode") or r.get("THSCODE") or r.get("code")
            if code is None:
                continue
            entry = out.setdefault(code, {})
            for k, v in r.items():
                if k.lower() == "thscode":
                    continue
                entry[k] = v
    return out


def _parse_edb_payload(
    payload: dict, indicators_map: dict[str, str]
) -> dict[str, list[dict]]:
    """edb 返回兼容解析。indicators_map 是 {别名: 指标编码}。"""
    rows = (
        payload.get("tables")
        or payload.get("data")
        or payload.get("result")
        or []
    )
    code_to_alias = {v: k for k, v in indicators_map.items()}
    out: dict[str, list[dict]] = {alias: [] for alias in indicators_map.keys()}

    if isinstance(rows, dict):
        # 列式：{date:[...], M0001:[...], M0002:[...]}
        date_key = "time" if "time" in rows else ("date" if "date" in rows else None)
        if date_key:
            dates = rows[date_key]
            for col, values in rows.items():
                if col == date_key:
                    continue
                alias = code_to_alias.get(col, col)
                if not isinstance(values, list):
                    continue
                for d, v in zip(dates, values):
                    if v is None or v == "":
                        continue
                    out.setdefault(alias, []).append({"date": d, "value": v})
        return out

    if isinstance(rows, list):
        for r in rows:
            # iFinD 真实返回每个 table 自身就是一个 series:
            #   {"id": ["L001619084"], "time": [...], "value": [...]}
            # id 是 list（取首项），time/value 是平行数组（按时间序对齐）。
            ids = r.get("id") or r.get("indicator") or r.get("INDICATOR") or r.get("indicode")
            if isinstance(ids, list):
                code = ids[0] if ids else None
            else:
                code = ids
            alias = code_to_alias.get(code, code) if code else None
            if alias is None:
                continue
            times = r.get("time") or r.get("date") or r.get("TIME") or []
            values = r.get("value") or r.get("VALUE") or []
            if isinstance(times, list) and isinstance(values, list):
                # 数组 series 形式（iFinD 标准）
                out.setdefault(alias, []).extend(
                    {"date": d, "value": v}
                    for d, v in zip(times, values)
                    if v is not None and v != ""
                )
            else:
                # 兜底：旧式 {time: scalar, value: scalar}
                if values:
                    out.setdefault(alias, []).append({"date": times, "value": values})

    # iFinD 返回是降序（最新在前），统一排成升序便于 caller 用 series[-1] 拿最新。
    for alias_key in out:
        out[alias_key].sort(key=lambda x: str(x.get("date") or ""))
    return out


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

    # ---------- THS_BD: 基础数据 ----------

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def basic_data(
        self,
        codes: list[str] | str,
        indicators: list[str] | None = None,
        indi_params: list[str] | None = None,
    ) -> dict[str, dict]:
        """THS_BD 等价接口：批量取证券静态/基础字段。

        参数:
            codes:        证券代码（"00700.HK" 或 list）
            indicators:   字段名列表，默认为港股公司常用基础字段（见 DEFAULT_BASIC_DATA_INDICATORS_HK）
            indi_params:  与 indicators 等长的参数列表（部分字段需要参数，如截止日期）

        返回:  {ticker: {indicator_name: value}}
        """
        if not self.configured:
            return {}
        s = get_settings()
        if isinstance(codes, str):
            code_list = [codes]
        else:
            code_list = list(codes)
        codes_str = ",".join(code_list)
        indicators = indicators or DEFAULT_BASIC_DATA_INDICATORS_HK
        ind_str = ",".join(indicators)
        para_str = ";".join(indi_params or ["" for _ in indicators])

        body = {
            "codes": codes_str,
            "indipara": ind_str,
            "indiparams": para_str,
        }
        try:
            payload = self._post(s.ths_endpoint_basic_data, body)
        except Exception as e:
            logger.warning(f"basic_data failed: {e}")
            return {}

        return _parse_table_to_dict_by_code(payload, code_list, indicators)

    # ---------- THS_EDB: 经济数据库 ----------

    @disk_cache(ttl_seconds=12 * 3600, namespace="ths")
    def edb(
        self,
        indicators: dict[str, str] | list[str] | None = None,
        start_date: str = "2022-01-01",
        end_date: str | None = None,
    ) -> dict[str, list[dict]]:
        """THS_EDB 等价接口：批量取宏观/行业经济指标时间序列。

        参数:
            indicators:  {别名: 指标编码} 或 [指标编码,...]；不传则用港股宏观默认集合。
            start_date / end_date: YYYY-MM-DD

        返回: {别名/指标编码: [{"date":..., "value":...}, ...]}
        """
        if not self.configured:
            return {}
        s = get_settings()
        end_date = end_date or time.strftime("%Y-%m-%d")

        if indicators is None:
            indicators_map = DEFAULT_HK_MACRO_EDB_CODES
        elif isinstance(indicators, list):
            indicators_map = {code: code for code in indicators}
        else:
            indicators_map = indicators

        codes_str = ",".join(indicators_map.values())
        body = {
            "indicators": codes_str,
            "startdate": start_date,
            "enddate": end_date,
        }
        try:
            payload = self._post(s.ths_endpoint_edb, body)
        except Exception as e:
            logger.warning(f"edb failed: {e}")
            return {}

        return _parse_edb_payload(payload, indicators_map)

    # ---------- THS_DR: 研究报告 ----------

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def research_reports(
        self,
        codes: list[str] | str | None = None,
        industry: str | None = None,
        start_date: str = "2024-01-01",
        end_date: str | None = None,
        report_type: str | None = None,
        keyword: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """THS_DR 等价接口：拉取研究报告/行业研报列表。

        codes / industry 至少传一个。返回示例：
            [{"date": "2025-04-08", "title": "...", "broker": "...",
              "rating": "买入", "abstract": "...", "pdfURL": "..."}]
        """
        if not self.configured:
            return []
        s = get_settings()
        end_date = end_date or time.strftime("%Y-%m-%d")

        body: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
            "limit": limit,
        }
        if codes:
            body["codes"] = ",".join(codes) if isinstance(codes, list) else codes
        if industry:
            body["industry"] = industry
        if report_type:
            body["reportType"] = report_type
        if keyword:
            body["keyword"] = keyword

        try:
            payload = self._post(s.ths_endpoint_data_report, body)
        except Exception as e:
            logger.warning(f"research_reports failed: {e}")
            return []

        rows = (
            payload.get("tables")
            or payload.get("data")
            or payload.get("result")
            or []
        )
        return rows if isinstance(rows, list) else [rows]

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
