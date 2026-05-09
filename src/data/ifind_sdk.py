"""iFinD QuantAPI Python SDK 封装层。

SDK 安装方式: 同花顺 iFinD 客户端的 installiFinDPy.py 脚本会在 venv 的 site-packages
里写一个 iFinDPy.pth 指向 SDK 目录, Python 启动时自动加入 sys.path。

为什么不用 REST: REST 端点 (edb_service / report_query) 仅覆盖 EDB / 公告。
DR 数据报表 (p05310/p05309)、HistoryQuotes、DataPool 必须走 SDK。

港股代码格式约定:
- 项目内部 / cli 输入: 5 位字符串 (例: "02670"); 4 位时也接受 (例: "0700")
- iFinD thscode: 4 位 + ".HK" (例: "2670.HK", "0700.HK")
- 转换规则: 去掉所有前导 0 后再 zfill(4)，最后接 ".HK"
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from typing import Any

from loguru import logger

from config import get_settings


_cache_conn: sqlite3.Connection | None = None
_cache_conn_lock = threading.Lock()


def _cache_db() -> sqlite3.Connection:
    """SDK 专用缓存表，单例连接（避免每次 SDK 调用都 open+close）。

    单次 prefetch 涉及 ~4 次 SDK 调用 × N 个 peer → 之前每次都开新连接，
    现在共享一个 connection 显著降低 IO。
    """
    global _cache_conn
    if _cache_conn is not None:
        return _cache_conn
    with _cache_conn_lock:
        if _cache_conn is not None:
            return _cache_conn
        p = get_settings().cache_dir / "cache.sqlite"
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB, ts REAL)")
        conn.commit()
        _cache_conn = conn
        return conn


def _cache_get(key: str, ttl_seconds: int) -> Any:
    row = _cache_db().execute("SELECT v, ts FROM kv WHERE k=?", (key,)).fetchone()
    if row is None:
        return None
    v, ts = row
    if time.time() - ts > ttl_seconds:
        return None
    try:
        return pickle.loads(v)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    conn = _cache_db()
    conn.execute(
        "INSERT OR REPLACE INTO kv VALUES (?, ?, ?)",
        (key, pickle.dumps(value), time.time()),
    )
    conn.commit()


def _cache_key(*parts: Any) -> str:
    blob = json.dumps({"ifind_sdk": parts}, default=str, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

_login_lock = threading.Lock()
_logged_in: bool = False
# 把"未配置/未安装"提示从 warning 降到 info, 且全局只打印一次
# (一次 prefetch 涉及十几次 SDK 调用, 每次都 warn 会刷屏)
_unconfigured_warned: bool = False
_sdk_missing_warned: bool = False


def to_ths_hk_code(ticker: str) -> str:
    """5/4 位港股 ticker → iFinD thscode.

    规则:
    - 普通港股: 去前导 0, zfill(4) + ".HK" (例: "02432" → "2432.HK")
    - IPO 询价中副牌: 以 H 开头 (例: "H2254") → 直接 "H2254.HK"
    """
    s = ticker.upper().replace(".HK", "").strip()
    if s.startswith("H") and len(s) > 1 and s[1:].isdigit():
        return f"{s}.HK"
    digits = s.lstrip("0") or "0"
    return f"{digits.zfill(4)}.HK"


def verify_company_name(thscode: str, expected_name: str) -> tuple[bool, str | None]:
    """校验 iFinD 中 thscode 对应的 corp_short_name 是否与 expected_name 模糊匹配。

    Returns: (matched: bool, actual_name: str | None)
    matched=False 时, actual_name 是 iFinD 实际查到的公司名 (可用于警告).
    """
    data = get_basic_data(thscode, "corp_short_name", "")
    actual = (data.get("corp_short_name") or "").strip()
    if not actual:
        return (False, None)
    # 模糊匹配: 把"机器人""科技""股份""有限公司"等通用后缀去掉, 比较 2-3 字核心词
    import re
    suffix_pat = re.compile(r"(机器人|智能科技|科技|集团|控股|股份|有限公司|有限|国际|-W|-B|-P)+$")
    norm_actual = suffix_pat.sub("", actual)
    norm_expected = suffix_pat.sub("", expected_name.strip())
    # 任意一个包含另一个的核心 (>=2 字) 视为匹配
    if not norm_actual or not norm_expected:
        return (actual == expected_name, actual)
    matched = (
        norm_actual in norm_expected
        or norm_expected in norm_actual
        or norm_actual[:2] == norm_expected[:2]  # 头 2 字相同放宽
    )
    return (matched, actual)


def _ensure_login() -> bool:
    """惰性登录 iFinD; 全局单例, 多次调用安全。返回是否成功。

    "SDK 未安装" / "未配置账号" 这类预期跳过场景只 info-level 提示一次，
    避免一次 prefetch 在日志里刷屏。
    """
    global _logged_in, _unconfigured_warned, _sdk_missing_warned
    if _logged_in:
        return True
    with _login_lock:
        if _logged_in:
            return True
        try:
            from iFinDPy import THS_iFinDLogin
        except ImportError:
            if not _sdk_missing_warned:
                logger.info(
                    "iFindPy SDK 未安装, SDK 接口不可用 (REST 路径仍正常)。"
                    "如需 peers 财务/估值数据, 请用 installiFinDPy.py 装到当前 venv。"
                )
                _sdk_missing_warned = True
            return False
        # 优先读 pydantic-settings (会从 .env 加载), 兜底回退 os.environ
        s = get_settings()
        user = s.ifind_username or os.environ.get("IFIND_USERNAME", "")
        pwd = s.ifind_password or os.environ.get("IFIND_PASSWORD", "")
        if not user or not pwd:
            if not _unconfigured_warned:
                logger.info("IFIND_USERNAME / IFIND_PASSWORD 未配置, 跳过 SDK 登录")
                _unconfigured_warned = True
            return False
        ret = THS_iFinDLogin(user, pwd)
        # 0 = 成功, -201 = 已登录(也算成功)
        if ret in (0, -201):
            logger.info(f"iFindPy SDK 登录成功 (ret={ret}, user={user})")
            _logged_in = True
            return True
        logger.warning(f"iFindPy SDK 登录失败 ret={ret}")
        return False


_BD_TTL = 24 * 3600
_HQ_TTL = 12 * 3600


def get_basic_data(thscode: str, fields: str, params: str = "") -> dict[str, Any]:
    """单股基础数据 (THS_BD). 仅缓存成功结果, 失败时不入缓存以便下次重试。

    Args:
        thscode: iFinD 格式港股代码 (如 "2432.HK")
        fields: 分号分隔字段, 如 "corp_short_name;ipo_date;ths_ipo_price_global"
        params: 与 fields 对齐的参数串 (分号分隔), 如 ";;OC"
    """
    key = _cache_key("BD", thscode, fields, params)
    cached = _cache_get(key, _BD_TTL)
    if cached is not None:
        return cached
    if not _ensure_login():
        return {}
    from iFinDPy import THS_BD

    try:
        r = THS_BD(thscode, fields, params)
    except Exception as e:
        logger.warning(f"THS_BD({thscode}) 异常: {e}")
        return {}
    if r.errorcode != 0 or r.data is None or r.data.empty:
        logger.warning(f"THS_BD({thscode}) ec={r.errorcode} errmsg={r.errmsg!r}")
        # 登录会话过期时强制重登一次再试
        if r.errorcode == -1010:
            global _logged_in
            _logged_in = False
            if _ensure_login():
                try:
                    r = THS_BD(thscode, fields, params)
                except Exception:
                    return {}
                if r.errorcode != 0 or r.data is None or r.data.empty:
                    return {}
            else:
                return {}
        else:
            return {}
    row = r.data.iloc[0].to_dict()
    import pandas as pd  # 局部 import：只有真调 SDK 才需要 pandas
    cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    _cache_set(key, cleaned)
    return cleaned


def get_history_quotes(
    thscode: str,
    fields: str = "close,changeRatio,amount,turnoverRatio",
    sdate: str | None = None,
    edate: str | None = None,
    *,
    days_back: int = 180,
) -> list[dict]:
    """单股历史日 K (THS_HQ). catalog 验证: 第 3 个参数 jsonparam 留空字符串即可。

    sdate/edate 为 None 时:
      edate = today
      sdate = today - days_back 日历日 (默认 180 ~= 90 交易日 + buffer)
    """
    from datetime import date, timedelta
    if edate is None:
        edate = date.today().strftime("%Y-%m-%d")
    if sdate is None:
        sdate = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    key = _cache_key("HQ", thscode, fields, sdate, edate)
    cached = _cache_get(key, _HQ_TTL)
    if cached is not None:
        return cached
    if not _ensure_login():
        return []
    from iFinDPy import THS_HQ

    try:
        r = THS_HQ(thscode, fields, "", sdate, edate)
    except Exception as e:
        logger.warning(f"THS_HQ({thscode}) 异常: {e}")
        return []
    if r.errorcode != 0 or r.data is None or r.data.empty:
        logger.warning(f"THS_HQ({thscode}) ec={r.errorcode} errmsg={r.errmsg!r}")
        if r.errorcode == -1010:
            global _logged_in
            _logged_in = False
            if _ensure_login():
                try:
                    r = THS_HQ(thscode, fields, "", sdate, edate)
                except Exception:
                    return []
                if r.errorcode != 0 or r.data is None or r.data.empty:
                    return []
            else:
                return []
        else:
            return []
    rows = r.data.to_dict(orient="records")
    _cache_set(key, rows)
    return rows


def compute_peer_quote_summary(
    ticker: str,
    sdate: str,
    edate: str,
) -> dict[str, Any]:
    """给定港股 ticker, 拉近期 K 线并算 30/90 日涨跌幅 + 最新价。

    返回: {ticker, thscode, latest_close, return_30d, return_90d, latest_date}
    数据缺失字段为 None。
    """
    code = to_ths_hk_code(ticker)
    rows = get_history_quotes(code, "close,changeRatio,amount", sdate, edate)
    out: dict[str, Any] = {
        "ticker": ticker,
        "thscode": code,
        "latest_close": None,
        "latest_date": None,
        "return_30d": None,
        "return_90d": None,
        "rows": len(rows),
    }
    if not rows:
        return out
    # rows 已按时间升序（iFinD 默认正序）
    latest = rows[-1]
    out["latest_close"] = latest.get("close")
    out["latest_date"] = latest.get("time")
    if len(rows) >= 30 and rows[-30].get("close"):
        p30 = rows[-30]["close"]
        if out["latest_close"] and p30:
            out["return_30d"] = round((out["latest_close"] / p30 - 1) * 100, 2)
    if len(rows) >= 90 and rows[-90].get("close"):
        p90 = rows[-90]["close"]
        if out["latest_close"] and p90:
            out["return_90d"] = round((out["latest_close"] / p90 - 1) * 100, 2)
    return out


def get_peer_valuation_multiples(ticker: str, date: str = "") -> dict[str, Any]:
    """单股估值倍数 (THS_BD): PE-TTM / PB / PEG / PS-TTM / EV/EBITDA / EV/Sales.

    Args:
        date: YYYY-MM-DD; 默认空 = 今天。
    """
    from datetime import date as _date
    if not date:
        date = _date.today().strftime("%Y-%m-%d")
    code = to_ths_hk_code(ticker)
    fields = "pe_ttm;pb_latest;his_peg;ps_ttm;ev2_to_ebitda;ev_to_sales_ratio"
    # PEG 第三个参数是当前年(2026), 第四个是 103; 其他是 100
    params = (
        f"{date},100;{date},100;{date},2026,103;{date},100;{date};{date}"
    )
    data = get_basic_data(code, fields, params)
    return {
        "ticker": ticker,
        "thscode": code,
        "pe_ttm": data.get("pe_ttm"),
        "pb_latest": data.get("pb_latest"),
        "his_peg": data.get("his_peg"),
        "ps_ttm": data.get("ps_ttm"),
        "ev_ebitda": data.get("ev2_to_ebitda"),
        "ev_sales": data.get("ev_to_sales_ratio"),
    }


def _fundamentals_for_year(ticker: str, year: int) -> dict[str, Any]:
    code = to_ths_hk_code(ticker)
    eoy = f"{year}-12-31"
    fields = "total_oi;ni_attr_to_cs;gross_selling_rate;net_profit_margin_on_sales;ths_roe_hks"
    params = f"{eoy};{eoy},100,OC;{eoy},104;{eoy},104;{eoy},100"
    data = get_basic_data(code, fields, params)
    return {
        "ticker": ticker,
        "thscode": code,
        "year": year,
        "revenue": data.get("total_oi"),
        "net_profit": data.get("ni_attr_to_cs"),
        "gross_margin": data.get("gross_selling_rate"),
        "net_margin": data.get("net_profit_margin_on_sales"),
        "roe": data.get("ths_roe_hks"),
    }


def _default_target_fiscal_year() -> int:
    """根据当前日期推算"应该已披露的最近完整年报"年份。

    港股年报通常 4-6 月披露。简化规则:
      7 月起 → 上一年（如 2026-07 → 2025 年报）
      6 月底前 → 前年（如 2026-05 → 2024 年报，2025 多数还没发）
    """
    from datetime import date
    today = date.today()
    return today.year - 1 if today.month >= 7 else today.year - 2


def get_peer_fundamentals(
    ticker: str,
    year: int | None = None,
    *,
    fallback_years: int = 1,
) -> dict[str, Any]:
    """单股基本面 (THS_BD): 营收 / 净利润 / 毛利率 / 净利率 / ROE.

    Args:
        year: 报告期年份。None 时按 _default_target_fiscal_year() 自动推算。
        fallback_years: 主年份没数据时再回退几年（默认 1 = 再试一年）。

    自动 fallback 逻辑：先试主年份，若 revenue / net_profit 都缺，则回退到前一年。
    避免 5 月跑时 hardcoded year=2025 拿到全空。
    """
    if year is None:
        year = _default_target_fiscal_year()

    result = _fundamentals_for_year(ticker, year)
    if result.get("revenue") is not None or result.get("net_profit") is not None:
        return result
    # 关键字段空 → 试更早年份
    for delta in range(1, fallback_years + 1):
        fallback_year = year - delta
        logger.info(
            f"[SDK] {ticker} {year} 年报数据缺失, fallback 到 {fallback_year}"
        )
        result = _fundamentals_for_year(ticker, fallback_year)
        if result.get("revenue") is not None or result.get("net_profit") is not None:
            return result
    return result  # 仍然空就返空，让 caller 自行处理


def get_peer_ipo_summary(ticker: str) -> dict[str, Any]:
    """给定港股 ticker, 拉公司简称 + IPO 价 + 上市日期 + 首日开盘价 + 首日开盘涨跌。"""
    code = to_ths_hk_code(ticker)
    fields = (
        "corp_short_name;ipo_date;ths_ipo_price_global;"
        "ths_ipo_maxprice_global;open_on_debut_g"
    )
    params = ";;OC;;"
    data = get_basic_data(code, fields, params)
    ipo_price = data.get("ths_ipo_price_global")
    first_day_open = data.get("open_on_debut_g")
    # 首日开盘相对招股价涨跌% (招股价为 0 / None 时不算)
    first_day_open_return = None
    if isinstance(ipo_price, (int, float)) and ipo_price > 0 and isinstance(first_day_open, (int, float)):
        first_day_open_return = round((first_day_open / ipo_price - 1) * 100, 2)
    return {
        "ticker": ticker,
        "thscode": code,
        "name": data.get("corp_short_name"),
        "ipo_date": data.get("ipo_date"),
        "ipo_price": ipo_price,
        "ipo_max_price": data.get("ths_ipo_maxprice_global"),
        "first_day_open": first_day_open,
        "first_day_open_return": first_day_open_return,
    }


# ============================================================================
# 港股指数 / 投后股价 / 公告查询
# ============================================================================

# 常用港股指数 thscode (iFinD 格式: <code>.HI)
HK_INDICES = {
    "HSI": "HSI.HI",         # 恒生指数
    "HSCEI": "HSCEI.HI",     # 恒生中国企业指数 (H 股)
    "HSTECH": "HSTECH.HI",   # 恒生科技指数
    "HSCI": "HSCI.HI",       # 恒生综合指数
    "HSCCI": "HSCCI.HI",     # 恒生中国 25 指数
    "HSML": "HSML.HI",       # 恒生医疗保健指数
}


def get_index_summary(index_code: str = "HSI") -> dict[str, Any]:
    """港股指数实时点位 + 估值（替换 akshare 的 get_hsi_index）。

    Args:
        index_code: 'HSI' / 'HSCEI' / 'HSTECH' 等; 也接受完整 thscode 如 'HSI.HI'

    Returns:
        {index, thscode, latest_close, latest_date, change_30d, change_90d,
         pe_ttm, pb_latest}; 任何字段拿不到都为 None。
    """
    code = index_code if "." in index_code else HK_INDICES.get(index_code.upper(), f"{index_code}.HI")

    # 1. 近期 K 线 → 当前点位 + 30/90 日变动
    from datetime import date, timedelta
    edate = date.today().strftime("%Y-%m-%d")
    sdate = (date.today() - timedelta(days=140)).strftime("%Y-%m-%d")
    rows = get_history_quotes(code, "close", sdate, edate)
    out: dict[str, Any] = {
        "index": index_code, "thscode": code,
        "latest_close": None, "latest_date": None,
        "change_30d": None, "change_90d": None,
        "pe_ttm": None, "pb_latest": None,
    }
    if rows:
        latest = rows[-1]
        out["latest_close"] = latest.get("close")
        out["latest_date"] = latest.get("time")
        if len(rows) >= 30:
            p30 = rows[-30].get("close")
            if out["latest_close"] and p30:
                out["change_30d"] = round((out["latest_close"] / p30 - 1) * 100, 2)
        if len(rows) >= 90:
            p90 = rows[-90].get("close")
            if out["latest_close"] and p90:
                out["change_90d"] = round((out["latest_close"] / p90 - 1) * 100, 2)

    # 2. 估值倍数（iFinD 部分指数有 pe_ttm / pb_latest 字段）
    today_str = edate
    fields = "pe_ttm;pb_latest"
    params = f"{today_str},100;{today_str},100"
    val = get_basic_data(code, fields, params)
    out["pe_ttm"] = val.get("pe_ttm")
    out["pb_latest"] = val.get("pb_latest")
    return out


def get_indices_summary(codes: list[str] | None = None) -> list[dict]:
    """批量取多个港股指数。默认 HSI / HSCEI / HSTECH 三大指数。"""
    if codes is None:
        codes = ["HSI", "HSCEI", "HSTECH"]
    return [get_index_summary(c) for c in codes]


def compute_post_ipo_returns(
    ticker: str,
    ipo_price: float,
    listing_date: str | None = None,
) -> dict[str, Any]:
    """根据 ticker + 招股价 + 上市日期，自动算上市后 d1/d30/d90/d180/d365 收益率。

    Args:
        ticker: 港股代码（5 位）。已上市后 ticker 通常已正式（不是 H 副牌）
        ipo_price: 招股价 HKD
        listing_date: 上市日期 'YYYY-MM-DD'; None 时尝试从 iFinD 拉

    Returns:
        {
          listing_date, d1_close, d30_close, d90_close, d180_close, d365_close,
          d1_return, d30_return, d90_return, d180_return, d365_return,  (相对招股价)
          d1_open_return,  (首日开盘 vs 招股价)
          was_broken_d1, was_broken_d180,
          max_drawdown_in_d180_pct, min_price_in_d180,
          avg_daily_turnover_hkd_m_d180,
        }
        任何字段缺失为 None；listing_date 拉不到则只返回基础结构。
    """
    code = to_ths_hk_code(ticker)

    # 上市日期：优先用户给定 / 否则查 iFinD
    if listing_date is None:
        ipo_summary = get_peer_ipo_summary(ticker)
        listing_date = ipo_summary.get("ipo_date")
    if not listing_date:
        return {"error": f"无法确定 {ticker} 上市日期，需显式传 listing_date"}

    from datetime import datetime, timedelta
    try:
        ld = datetime.strptime(listing_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"上市日期格式异常: {listing_date}"}

    # 拉上市日 ~ 上市日+400 日的日 K + 成交额
    sdate = ld.strftime("%Y-%m-%d")
    edate = min(
        (ld + timedelta(days=400)).strftime("%Y-%m-%d"),
        datetime.today().strftime("%Y-%m-%d"),
    )
    rows = get_history_quotes(code, "open,close,amount", sdate, edate)

    out: dict[str, Any] = {
        "ticker": ticker, "thscode": code, "listing_date": listing_date,
        "ipo_price": ipo_price,
        "d1_close": None, "d1_return": None, "d1_open_return": None,
        "d30_close": None, "d30_return": None,
        "d90_close": None, "d90_return": None,
        "d180_close": None, "d180_return": None,
        "d365_close": None, "d365_return": None,
        "was_broken_d1": None, "was_broken_d180": None,
        "max_drawdown_in_d180_pct": None,
        "min_price_in_d180": None,
        "avg_daily_turnover_hkd_m_d180": None,
    }
    if not rows:
        return out

    # rows 升序（_parse_edb_payload 已 sort，get_history_quotes 默认升序）
    def _close_at(idx: int) -> float | None:
        if 0 <= idx < len(rows):
            v = rows[idx].get("close")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _ret(close: float | None) -> float | None:
        if close is None or ipo_price <= 0:
            return None
        return round(close / ipo_price - 1, 4)

    # rows[0] 是首日; 用 1, 30, 90, 180, 365 个交易日（不是日历日）
    out["d1_close"] = _close_at(0)
    out["d30_close"] = _close_at(min(29, len(rows) - 1))
    out["d90_close"] = _close_at(min(89, len(rows) - 1))
    out["d180_close"] = _close_at(min(179, len(rows) - 1))
    out["d365_close"] = _close_at(min(244, len(rows) - 1))  # 港股年交易日 ~245
    out["d1_return"] = _ret(out["d1_close"])
    out["d30_return"] = _ret(out["d30_close"])
    out["d90_return"] = _ret(out["d90_close"])
    out["d180_return"] = _ret(out["d180_close"])
    out["d365_return"] = _ret(out["d365_close"])

    # 首日开盘 vs 招股价
    first_open = rows[0].get("open")
    try:
        first_open_f = float(first_open) if first_open is not None else None
    except (TypeError, ValueError):
        first_open_f = None
    if first_open_f is not None and ipo_price > 0:
        out["d1_open_return"] = round(first_open_f / ipo_price - 1, 4)

    # 是否破发
    if out["d1_close"] is not None:
        out["was_broken_d1"] = out["d1_close"] < ipo_price
    if out["d180_close"] is not None:
        out["was_broken_d180"] = out["d180_close"] < ipo_price

    # 锁定期内最大回撤、最低价
    closes_180 = [_close_at(i) for i in range(min(180, len(rows)))]
    closes_180 = [c for c in closes_180 if c is not None]
    if closes_180:
        peak = ipo_price
        max_dd = 0.0
        for c in closes_180:
            if c > peak:
                peak = c
            dd = (c - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
        out["max_drawdown_in_d180_pct"] = round(max_dd * 100, 2)
        out["min_price_in_d180"] = round(min(closes_180), 4)

    # 180 日均成交额（港币百万）—— iFinD amount 单位通常是港元，转成 M
    amounts = []
    for i in range(min(180, len(rows))):
        a = rows[i].get("amount")
        try:
            if a is not None:
                amounts.append(float(a))
        except (TypeError, ValueError):
            pass
    if amounts:
        out["avg_daily_turnover_hkd_m_d180"] = round(sum(amounts) / len(amounts) / 1_000_000, 2)

    return out


def get_recent_announcements(
    ticker: str,
    days: int = 180,
    limit: int = 20,
) -> list[dict]:
    """拉某 ticker 近 N 天的临时公告标题列表（依托 ths_client.report_query）。

    用途: peers 近期减持/业绩公告 → sentiment/risk 看 peer 是否有"实际控制人减持"
    "盈利预警"等会影响赛道情绪的事件。

    Args:
        ticker: 港股 5/4 位代码
        days: 回看天数（默认 180）
        limit: 返回条数上限

    Returns: [{date, title, pdfURL, reportType}, ...]
    """
    from datetime import date, timedelta

    from src.data.ths_client import THSClient

    client = THSClient()
    if not client.configured:
        logger.debug("THS REST 未配置, 跳过 announcements 查询")
        return []

    code = to_ths_hk_code(ticker)
    edate = date.today().strftime("%Y-%m-%d")
    sdate = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    rows = client.query_reports(
        codes=code,
        function_params={"startDate": sdate, "endDate": edate},
    )
    out = []
    for r in rows[:limit]:
        out.append({
            "date": r.get("declareDate") or r.get("DECLAREDATE") or r.get("publishTime"),
            "title": r.get("title") or r.get("TITLE", ""),
            "pdfURL": r.get("pdfURL") or r.get("PDFURL", ""),
            "reportType": r.get("reportType") or r.get("REPORTTYPE", ""),
        })
    return out


# 用关键词标注公告类型（便于 sentiment / risk 直接看到是否有"减持/业绩预警"）
_ANNOUNCEMENT_TAGS = {
    "减持": ["减持", "出售股份", "出售股票", "Disposal of Shares"],
    "业绩": ["业绩", "盈利预警", "Profit Warning", "中期业绩", "年度业绩"],
    "回购": ["回购", "Buy-back", "Share Repurchase"],
    "增发": ["配售", "增发", "Placing", "Top-up"],
    "更名": ["更改公司名称", "Change of Company Name"],
    "调整": ["调整", "重组", "Reorganization"],
}


def tag_announcement(title: str) -> list[str]:
    """从公告标题里抽出标签（减持/业绩/回购等）。"""
    title_low = title.lower()
    out = []
    for tag, kws in _ANNOUNCEMENT_TAGS.items():
        if any(kw.lower() in title_low for kw in kws):
            out.append(tag)
    return out
