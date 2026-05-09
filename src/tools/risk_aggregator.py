"""RiskAggregator — 综合风险等级的确定性公式。

设计动机:
之前 Risk Agent 8 维评分后让 LLM 给"综合 1-5 级", LLM 经常给一个比加权平均
更高/更低的"主观感觉值", 不可解释. 本模块用确定性公式聚合, LLM 只输出每维度
得分, 综合等级由 Python 算.

聚合规则 (基石投资专属):
1. 加权平均 - 各维度按重要性加权
2. max() 约束 - 任一维度极高风险 (≤1.5) 时综合不能超过 2.0
3. veto count - veto_conditions 触发时综合至多 2.0
4. 一致性放大 - 多个维度都低分时 (3+ 个 ≤2.5) 综合再 -0.5

返回值含解释:
- overall_risk_level (1-5)
- contributing_factors (哪些维度拉高/拉低综合)
- aggregation_method (用了哪条规则)
- warnings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 8 维风险维度的默认权重 (基石投资专属)
DEFAULT_RISK_WEIGHTS = {
    "信用与财务造假": 0.20,    # 财务问题致命, 权重最高
    "估值高估与破发": 0.18,    # 6 月禁售期内破发是核心痛点
    "行业逆风": 0.13,
    "流动性": 0.12,            # 解禁退出的关键
    "股东减持/解禁压力": 0.10,
    "监管与合规": 0.10,
    "关连交易与公司治理": 0.10,
    "ESG": 0.07,
}


@dataclass
class RiskItemInput:
    """LLM 提供的单一风险维度评分."""
    dimension: str
    score: float                         # 1-5 (1=极高风险, 5=极低)
    probability: str = "中"               # 极低 / 低 / 中 / 高
    impact: str = "中"                    # 小 / 中 / 大 / 极大
    early_warning: list[str] = field(default_factory=list)
    weight_override: float | None = None  # 显式指定权重, 覆盖默认


@dataclass
class RiskAggregateResult:
    overall_risk_level: float       # 加权综合后的最终等级 (1-5)
    raw_weighted_avg: float          # 原始加权平均 (无 max 约束)
    contributing_factors: list[dict]  # 每维度对综合的贡献度
    aggregation_method: str          # "weighted_avg" / "veto_capped" / "consensus_low"
    warnings: list[str]


def aggregate_risk_level(
    items: list[RiskItemInput],
    *,
    veto_count: int = 0,
    weights_override: dict[str, float] | None = None,
) -> RiskAggregateResult:
    """聚合多维度风险得分到综合等级 (1-5)。

    Args:
        items: LLM 提供的各维度评分列表
        veto_count: 风控 Agent 已识别的 veto 条件数
        weights_override: 自定义权重 (e.g. ListingProfile 调整)

    Returns: RiskAggregateResult, 含综合等级 + 解释 + warnings
    """
    if not items:
        return RiskAggregateResult(
            overall_risk_level=3.0, raw_weighted_avg=3.0,
            contributing_factors=[], aggregation_method="empty",
            warnings=["无风险维度评分输入, 默认 3.0"],
        )

    weights_map = dict(DEFAULT_RISK_WEIGHTS)
    if weights_override:
        weights_map.update(weights_override)

    contributions: list[dict] = []
    weighted_sum = 0.0
    sum_weights = 0.0
    for item in items:
        w = item.weight_override if item.weight_override is not None else weights_map.get(
            item.dimension, 1.0 / len(items)  # 未知维度均权
        )
        weighted_sum += item.score * w
        sum_weights += w
        contributions.append({
            "dimension": item.dimension,
            "score": item.score,
            "weight": w,
            "contribution": round(item.score * w, 3),
        })

    if sum_weights <= 0:
        sum_weights = float(len(items))

    raw_avg = weighted_sum / sum_weights
    final = raw_avg
    method = "weighted_avg"
    warnings: list[str] = []

    # 规则 1: 任一维度极高风险 (score ≤ 1.5) 时综合不超过 2.0
    extreme_low = [item for item in items if item.score <= 1.5]
    if extreme_low:
        cap = 2.0
        if final > cap:
            method = "extreme_low_capped"
            warnings.append(
                f"维度 [{', '.join(e.dimension for e in extreme_low)}] 评分 ≤ 1.5 极高风险, "
                f"综合等级被压制到 ≤ {cap} (原 {raw_avg:.2f})"
            )
            final = cap

    # 规则 2: veto_count >= 1 时综合不超过 2.0
    if veto_count >= 1 and final > 2.0:
        method = "veto_capped"
        warnings.append(
            f"风控触发 {veto_count} 个 veto 条件, 综合等级被压制到 ≤ 2.0 (原 {raw_avg:.2f})"
        )
        final = 2.0

    # 规则 3: 多个维度普遍低分 (≥3 个 score ≤ 2.5) 时再 -0.5
    low_count = sum(1 for item in items if item.score <= 2.5)
    if low_count >= 3:
        if method == "weighted_avg":
            method = "consensus_low"
        warnings.append(
            f"{low_count} 个维度 score ≤ 2.5 (3+ 维度低分共识), 综合等级 -0.5"
        )
        final = max(1.0, final - 0.5)

    # Clamp 到 [1, 5]
    final = max(1.0, min(5.0, final))

    return RiskAggregateResult(
        overall_risk_level=round(final, 2),
        raw_weighted_avg=round(raw_avg, 2),
        contributing_factors=contributions,
        aggregation_method=method,
        warnings=warnings,
    )


def explain_risk_aggregation(result: RiskAggregateResult) -> str:
    """渲染聚合结果为人类可读的解释 (供 markdown / HTML 用)."""
    lines = [
        f"**综合风险等级: {result.overall_risk_level} / 5**",
        f"原始加权平均: {result.raw_weighted_avg}, 聚合规则: {result.aggregation_method}",
        "",
    ]
    if result.warnings:
        lines.append("调整说明:")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")
    if result.contributing_factors:
        lines.append("各维度贡献:")
        sorted_contribs = sorted(
            result.contributing_factors,
            key=lambda x: x["contribution"], reverse=True,
        )
        for c in sorted_contribs:
            lines.append(
                f"- {c['dimension']}: score {c['score']} × weight {c['weight']:.2f} "
                f"= {c['contribution']}"
            )
    return "\n".join(lines)
