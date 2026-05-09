"""确定性的财务指标计算工具。LLM 不擅长算数，凡是能算的都在这里算好再喂进去。"""
from __future__ import annotations

from typing import Any


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def cagr(values: list[float], years: int) -> float | None:
    if not values or values[0] <= 0 or values[-1] <= 0 or years <= 0:
        return None
    return (values[-1] / values[0]) ** (1 / years) - 1


def summarize_income_trend(income_records: list[dict]) -> dict[str, Any]:
    """从 akshare 风格的利润表记录提炼三年营收/净利润/毛利率趋势。

    适配字段名差异：常见 '营业总收入'/'营业收入', '归属于母公司股东净利润'/'归属母公司股东净利润'。
    """
    if not income_records:
        return {}

    def pick(rec: dict, *keys: str) -> float | None:
        for k in keys:
            if k in rec and rec[k] not in (None, ""):
                try:
                    return float(rec[k])
                except (ValueError, TypeError):
                    continue
        return None

    sorted_recs = sorted(income_records, key=lambda r: str(r.get("报告期", r.get("REPORT_DATE", ""))))
    revenues = [pick(r, "营业总收入", "营业收入", "REVENUE") for r in sorted_recs]
    net_profits = [pick(r, "归属母公司股东净利润", "归属于母公司股东净利润", "NET_PROFIT") for r in sorted_recs]
    revenues = [v for v in revenues if v is not None]
    net_profits = [v for v in net_profits if v is not None]

    return {
        "periods": [r.get("报告期") for r in sorted_recs],
        "revenue": revenues,
        "net_profit": net_profits,
        "revenue_cagr": cagr(revenues, max(len(revenues) - 1, 1)) if len(revenues) >= 2 else None,
        "profit_cagr": cagr(net_profits, max(len(net_profits) - 1, 1)) if len(net_profits) >= 2 else None,
        "latest_net_margin": safe_div(net_profits[-1] if net_profits else None,
                                      revenues[-1] if revenues else None),
    }
