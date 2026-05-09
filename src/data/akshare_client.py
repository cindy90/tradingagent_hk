"""akshare 港股数据封装。所有函数返回纯 dict / list，便于序列化和缓存。

港股代码格式：5 位数字，例如腾讯 "00700"。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from .cache import disk_cache


def _import_akshare():
    try:
        import akshare as ak  # noqa: F401
        return ak
    except Exception as e:
        logger.warning(f"akshare 未安装或加载失败: {e}")
        return None


@disk_cache(ttl_seconds=3600, namespace="ak")
def get_hk_quote(ticker: str) -> dict[str, Any]:
    """获取港股实时报价。"""
    ak = _import_akshare()
    if ak is None:
        return {}
    try:
        df = ak.stock_hk_spot_em()
        row = df[df["代码"] == ticker]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    except Exception as e:
        logger.warning(f"get_hk_quote({ticker}) failed: {e}")
        return {}


@disk_cache(ttl_seconds=24 * 3600, namespace="ak")
def get_hk_hist(ticker: str, start: str, end: str, adjust: str = "qfq") -> list[dict]:
    """历史日线，start/end 格式 YYYYMMDD。"""
    ak = _import_akshare()
    if ak is None:
        return []
    try:
        df = ak.stock_hk_hist(symbol=ticker, period="daily", start_date=start, end_date=end, adjust=adjust)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"get_hk_hist({ticker}) failed: {e}")
        return []


@disk_cache(ttl_seconds=7 * 24 * 3600, namespace="ak")
def get_hk_financials(ticker: str) -> dict[str, Any]:
    """获取港股财务指标摘要（不同 akshare 版本接口可能不同，按需调整）。"""
    ak = _import_akshare()
    if ak is None:
        return {}
    out: dict[str, Any] = {"ticker": ticker}
    for name, fn_name in [
        ("income", "stock_hk_financial_em_income"),
        ("balance", "stock_hk_financial_em_balance"),
        ("cashflow", "stock_hk_financial_em_cashflow"),
    ]:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn(symbol=ticker)
            out[name] = df.to_dict(orient="records") if df is not None else []
        except Exception as e:
            logger.warning(f"akshare.{fn_name}({ticker}) failed: {e}")
            out[name] = []
    return out


@disk_cache(ttl_seconds=24 * 3600, namespace="ak")
def get_hk_industry_peers(industry_keyword: str) -> list[dict]:
    """按行业关键字粗筛港股可比公司列表。"""
    ak = _import_akshare()
    if ak is None:
        return []
    try:
        df = ak.stock_hk_spot_em()
        mask = df["名称"].str.contains(industry_keyword, na=False, regex=False)
        return df[mask][["代码", "名称", "最新价", "涨跌幅", "成交额"]].to_dict(orient="records")
    except Exception as e:
        logger.warning(f"get_hk_industry_peers failed: {e}")
        return []


@disk_cache(ttl_seconds=3600, namespace="ak")
def get_hsi_index() -> dict[str, Any]:
    """恒生指数当前点位/涨跌。"""
    ak = _import_akshare()
    if ak is None:
        return {}
    try:
        df = ak.stock_hk_index_spot_em()
        row = df[df["代码"] == "HSI"]
        if not row.empty:
            return row.iloc[0].to_dict()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        logger.warning(f"get_hsi_index failed: {e}")
        return {}
