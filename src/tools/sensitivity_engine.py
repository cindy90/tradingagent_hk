"""SensitivityEngine — 三档情景敏感性自动生成。

设计动机:
之前 LLM 猜悲观/基准/乐观三档估值, 数字常常自相矛盾或概率合不到 1.
本引擎给定 base case 假设 (毛利率 / CAGR / PS 倍数 / 等), 用确定性的
敏感度方程自动生成三档:
- 悲观: 核心假设 -2σ → 估值压缩
- 基准: base case
- 乐观: 核心假设 +1σ → 估值扩张

LLM 只负责:
- 提供 base case 假设值
- 描述 trigger 条件文本
- 概率赋值 + rationale (创造性)

引擎做:
- 应用敏感度公式
- 概率归一化 (强制三档之和 = 1.0)
- 生成 expected_return_pct (以 valuation_at_ipo 为基准反推)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseCase:
    """基准情景的核心假设. LLM 提供, 引擎应用敏感度。"""
    valuation_hkd_b: float          # 基准估值 (亿 HKD)
    revenue_cagr: float | None = None  # 0.35 = 35%
    gross_margin: float | None = None  # 0.38 = 38%
    ps_multiple: float | None = None   # 22x
    pe_multiple: float | None = None   # 70x
    # 用于反推回报率
    valuation_at_ipo_hkd_b: float | None = None  # 招股价对应市值 (用于算 return)


@dataclass
class ScenarioRowComputed:
    """三档情景的某一行 (确定性引擎输出). 与 decision.ScenarioRow 字段对齐。"""
    name: str   # 悲观 / 基准 / 乐观
    triggers: list[str]
    valuation_hkd_b: float
    probability: float
    expected_return_pct: float | None
    valuation_derivation: str
    probability_rationale: str


# ============================ 敏感度规则 ============================

@dataclass
class SensitivityShock:
    """每个核心假设在悲观/乐观情景下的扰动幅度。"""
    pessimistic_pct: float   # -25% 表示该指标在悲观下降 25%
    optimistic_pct: float    # +15% 表示乐观下上升 15%
    impact_on_valuation: float  # 0.7 表示该假设变动 1% 估值变动 0.7% (弹性系数)


# 不同 ListingProfile 下的标准敏感度规则
# (LLM 也可以 override, 但这里给一套合理默认)
DEFAULT_SHOCKS = {
    "revenue_cagr": SensitivityShock(-0.40, 0.30, 0.85),
        # CAGR -40% (35% → 21%) 估值压缩 ~34%; +30% (35% → 45%) 估值扩张 ~25%
    "gross_margin": SensitivityShock(-0.20, 0.10, 0.50),
        # 毛利 -20% (38% → 30%) 估值压缩 10%
    "ps_multiple": SensitivityShock(-0.25, 0.15, 1.0),
        # PS 直接线性影响估值
    "pe_multiple": SensitivityShock(-0.30, 0.20, 1.0),
}


# 不同 listing_chapter 下的扰动放大系数 (波动率差异)
PROFILE_VOLATILITY_MULTIPLIER = {
    "Main_Board_18A": 1.4,  # 18A 临床失败影响极大
    "Main_Board_18C": 1.3,  # 18C 技术兑现波动大
    "Main_Board_19C": 1.0,
    "Secondary_Listing": 1.1,
    "Dual_Primary_AH": 0.85,  # AH 估值更稳
    "Main_Board_Standard": 1.0,
    "GEM": 1.5,
    "Unknown": 1.0,
}


# 不同 size_tier 的概率默认值 (小盘破发率高 → 悲观概率上调)
DEFAULT_PROBABILITIES_BY_SIZE = {
    "Small": (0.40, 0.45, 0.15),   # 悲观 / 基准 / 乐观
    "Mid":   (0.30, 0.50, 0.20),
    "Large": (0.25, 0.55, 0.20),
    "Mega":  (0.20, 0.60, 0.20),
    "Unknown": (0.30, 0.50, 0.20),
}


def _apply_shock_to_valuation(
    base_val: float,
    shock_pct: float,
    elasticity: float,
) -> float:
    """假设 X 变动 shock_pct, 估值变动 = base × shock_pct × elasticity."""
    return base_val * (1 + shock_pct * elasticity)


def compute_sensitivity_table(
    base: BaseCase,
    *,
    listing_chapter: str = "Unknown",
    size_tier: str = "Unknown",
    pessimistic_triggers: list[str] | None = None,
    optimistic_triggers: list[str] | None = None,
    probability_rationale_pessimistic: str = "",
    probability_rationale_base: str = "",
    probability_rationale_optimistic: str = "",
    custom_probabilities: tuple[float, float, float] | None = None,
) -> list[ScenarioRowComputed]:
    """生成三档敏感性情景行 (悲观 / 基准 / 乐观)。

    LLM 输入: base + triggers + rationale (可选 custom_probabilities)
    引擎输出: 估值 / 回报率 / probability 自动算
    """
    vol_mult = PROFILE_VOLATILITY_MULTIPLIER.get(listing_chapter, 1.0)

    # 计算悲观/乐观估值 (按 base 值 + 主导假设的扰动 × volatility multiplier)
    # 简化: 用最敏感的核心假设 (有 ps_multiple 用 ps; 否则 cagr)
    if base.ps_multiple is not None:
        primary = "ps_multiple"
    elif base.pe_multiple is not None:
        primary = "pe_multiple"
    elif base.revenue_cagr is not None:
        primary = "revenue_cagr"
    else:
        primary = "gross_margin"

    shock = DEFAULT_SHOCKS.get(primary, DEFAULT_SHOCKS["ps_multiple"])
    pess_shock = shock.pessimistic_pct * vol_mult
    opt_shock = shock.optimistic_pct * vol_mult

    pess_val = round(_apply_shock_to_valuation(base.valuation_hkd_b, pess_shock, shock.impact_on_valuation), 2)
    opt_val = round(_apply_shock_to_valuation(base.valuation_hkd_b, opt_shock, shock.impact_on_valuation), 2)

    # 概率 (用户给 custom_probabilities 优先, 否则按 size_tier 默认)
    if custom_probabilities and abs(sum(custom_probabilities) - 1.0) <= 0.05:
        p_pess, p_base, p_opt = custom_probabilities
    else:
        p_pess, p_base, p_opt = DEFAULT_PROBABILITIES_BY_SIZE.get(
            size_tier, DEFAULT_PROBABILITIES_BY_SIZE["Unknown"])

    # 归一化 (确保 sum = 1.0)
    s = p_pess + p_base + p_opt
    if s > 0 and abs(s - 1.0) > 0.001:
        p_pess, p_base, p_opt = p_pess / s, p_base / s, p_opt / s

    # 回报率 (相对 valuation_at_ipo, 缺失时按 base mid 算)
    ipo_anchor = base.valuation_at_ipo_hkd_b or base.valuation_hkd_b
    if ipo_anchor and ipo_anchor > 0:
        ret_pess = round((pess_val / ipo_anchor - 1) * 100, 2)
        ret_base = round((base.valuation_hkd_b / ipo_anchor - 1) * 100, 2)
        ret_opt = round((opt_val / ipo_anchor - 1) * 100, 2)
    else:
        ret_pess = ret_base = ret_opt = None

    # 推算路径文字
    derivation_pess = (
        f"{primary} 悲观扰动 {pess_shock*100:+.0f}% × 弹性 {shock.impact_on_valuation} = "
        f"估值 {base.valuation_hkd_b} → {pess_val} 亿 HKD"
    )
    derivation_base = f"基准估值 {base.valuation_hkd_b} 亿 HKD (LLM 提供 base case)"
    derivation_opt = (
        f"{primary} 乐观扰动 {opt_shock*100:+.0f}% × 弹性 {shock.impact_on_valuation} = "
        f"估值 {base.valuation_hkd_b} → {opt_val} 亿 HKD"
    )

    return [
        ScenarioRowComputed(
            name="悲观",
            triggers=pessimistic_triggers or [f"{primary} 大幅恶化 ({pess_shock*100:+.0f}%)"],
            valuation_hkd_b=pess_val,
            probability=round(p_pess, 4),
            expected_return_pct=ret_pess,
            valuation_derivation=derivation_pess,
            probability_rationale=probability_rationale_pessimistic
                or f"size_tier={size_tier} 默认悲观概率 {p_pess:.0%}",
        ),
        ScenarioRowComputed(
            name="基准",
            triggers=[f"core assumption 持平 ({primary} 维持 base case)"],
            valuation_hkd_b=base.valuation_hkd_b,
            probability=round(p_base, 4),
            expected_return_pct=ret_base,
            valuation_derivation=derivation_base,
            probability_rationale=probability_rationale_base
                or f"size_tier={size_tier} 默认基准概率 {p_base:.0%}",
        ),
        ScenarioRowComputed(
            name="乐观",
            triggers=optimistic_triggers or [f"{primary} 显著改善 ({opt_shock*100:+.0f}%)"],
            valuation_hkd_b=opt_val,
            probability=round(p_opt, 4),
            expected_return_pct=ret_opt,
            valuation_derivation=derivation_opt,
            probability_rationale=probability_rationale_optimistic
                or f"size_tier={size_tier} 默认乐观概率 {p_opt:.0%}",
        ),
    ]
