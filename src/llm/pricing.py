"""模型计费表（CNY/百万 token）。每年更新。

主要用于在 token 账本里给出"本次预估成本"，不参与计费校验。
价格变动时只需要改这一处。
"""
from __future__ import annotations

# 单位: CNY / 1M token
# 来源: 各家公开定价（2026-05 quote）
# 缺失时按相近模型估算，标 _ESTIMATED 注释提醒
PRICING_CNY_PER_M_TOKEN: dict[str, dict[str, float]] = {
    # Anthropic（按官方定价 USD × 7.2 兑率折算）
    "claude-haiku-4-5-20251001": {"input": 6.0, "output": 30.0, "cache_read": 0.6},
    "claude-sonnet-4-6": {"input": 22.0, "output": 108.0, "cache_read": 2.2},
    "claude-opus-4-7": {"input": 108.0, "output": 540.0, "cache_read": 10.8},
    # Kimi (Moonshot)
    "moonshot-v1-8k": {"input": 12.0, "output": 12.0, "cache_read": 0},
    "moonshot-v1-32k": {"input": 24.0, "output": 24.0, "cache_read": 0},
    "moonshot-v1-128k": {"input": 60.0, "output": 60.0, "cache_read": 0},
    "kimi-latest": {"input": 12.0, "output": 12.0, "cache_read": 0},
    # DeepSeek
    "deepseek-chat": {"input": 1.0, "output": 8.0, "cache_read": 0.1},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0, "cache_read": 0.4},
}


def estimate_cost_cny(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """单次调用预估成本，CNY。"""
    p = PRICING_CNY_PER_M_TOKEN.get(model)
    if p is None:
        return 0.0
    cost = (
        (input_tokens - cache_read_tokens) * p["input"]
        + cache_read_tokens * p.get("cache_read", p["input"])
        + output_tokens * p["output"]
    ) / 1_000_000
    return round(cost, 4)


def estimate_total_cost_cny(ledger_by_tier: dict[str, dict], tier_to_model: dict[str, str]) -> float:
    total = 0.0
    for tier, slot in ledger_by_tier.items():
        model = tier_to_model.get(tier)
        if not model:
            continue
        total += estimate_cost_cny(
            model, slot.get("input", 0), slot.get("output", 0), slot.get("cache_read", 0)
        )
    return round(total, 2)
