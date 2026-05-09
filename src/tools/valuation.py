"""可比公司估值工具。计算 P/E、P/S、EV/EBITDA 区间并给出锚定估值。"""
from __future__ import annotations

import statistics
from typing import Any


def percentile(data: list[float], p: float) -> float | None:
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def comparable_valuation(
    target_metric: float,
    peer_multiples: list[float],
    metric_name: str = "PE",
) -> dict[str, Any]:
    """给定目标公司单期指标（如净利润）与可比公司估值倍数，输出估值区间。

    返回估值锚点：P25 / 中位数 / P75，均按 倍数 × 目标指标 计算。
    """
    cleaned = [m for m in peer_multiples if m is not None and m > 0]
    if not cleaned or target_metric is None:
        return {}

    mid = statistics.median(cleaned)
    p25 = percentile(cleaned, 0.25)
    p75 = percentile(cleaned, 0.75)

    return {
        "metric_name": metric_name,
        "target_metric": target_metric,
        "peer_count": len(cleaned),
        "multiple_p25": p25,
        "multiple_median": mid,
        "multiple_p75": p75,
        "valuation_low": p25 * target_metric if p25 else None,
        "valuation_mid": mid * target_metric,
        "valuation_high": p75 * target_metric if p75 else None,
    }
