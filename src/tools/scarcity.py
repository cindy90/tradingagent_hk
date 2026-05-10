"""ScarcityEngine — 港股稀缺性的确定性量化。

设计动机:
基石投资里"稀缺性 = 议价权 + 估值溢价 + 解禁后流动性"。一家"独苗"赛道公司
和一家"红海赛道又来一个的"在情绪 / 估值 / 流动性表现上完全不同。
LLM 凭印象判稀缺度容易跑偏 (训练记忆里的港股结构早过时), 因此把"算"挪到 Python:

输入: peers (来自 PeerSuggester + iFinD GICS 检索) + target ticker
输出: 量化指标 + 1-5 分稀缺度 (5=独苗 / 1=红海)

5 个核心指标:
1. listed_count_in_theme: 同主题已上市公司数量 (含 target 自己外的 peer 数)
2. liquidity_thinning_ratio: 同主题中 30 日均成交 < 5000 万 HKD 的占比
   → 高占比 = 即使"稀缺"也是流动性枯竭型, 不构成真稀缺
3. market_cap_concentration: 同主题中 top 3 市值占总市值比例
   → 高 = 龙头垄断, 中小盘被边缘化
4. avg_first_day_return: 同主题最近 IPO 的首日均值表现
   → 衡量市场对这赛道的最新追捧度 (≠ scarcity 本身, 但联动)
5. recent_ipo_density: 过去 12 个月同主题 IPO 数量
   → 高 = 赛道近期已被反复发, 边际稀缺度递减

LLM 拿这些指标做定性论证 + 给出最终 1-5 分。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


@dataclass
class ScarcityStats:
    """同主题港股稀缺性的可审计统计."""
    listed_count_in_theme: int          # 已上市同主题公司数 (不含 target 自身)
    market_cap_total_hkd_b: float       # 同主题总市值 (亿 HKD)
    market_cap_top3_share: float | None # top 3 市值占总市值
    median_market_cap_hkd_b: float | None
    liquidity_thinning_count: int       # 30 日均成交 < 5000 万 的家数
    liquidity_thinning_ratio: float | None  # 占比 (0-1)
    avg_first_day_return_pct: float | None  # 同主题最近 IPO 首日均值
    recent_ipo_count_12m: int           # 过去 12 月同主题 IPO 数
    # v2 流量稀缺度
    pipeline_count_in_theme: int = 0    # 同主题排队中 (申请版本/已通过聆讯) 公司数
    pipeline_companies: list[str] = field(default_factory=list)
                                         # 排队中同主题公司名 (用于 prompt 透明度)
    raw_scarcity_score: float = 3.0     # 引擎初算分 (1-5, LLM 可微调)
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compute_market_cap_hkd_b(peer: dict) -> float | None:
    """从 peer dict 抽市值 (亿 HKD). 兼容多种字段名."""
    for key in ("market_cap_hkd_b", "market_cap_hkd", "total_mv_hkd", "mv_hkd_b"):
        v = _safe_float(peer.get(key))
        if v is not None:
            # 字段以"亿"为单位时直接用; 以原始 HKD 为单位时换算
            if key.endswith("_b") or v < 1e4:
                return v
            return v / 1e8
    # 退化: market_cap × 汇率
    cap_rmb = _safe_float(peer.get("total_mv"))
    if cap_rmb is not None and cap_rmb > 0:
        return cap_rmb * 1.10 / 1e8
    return None


def _is_recent_ipo(ipo_date_str: Any, cutoff_days: int = 365) -> bool:
    if not ipo_date_str:
        return False
    try:
        if isinstance(ipo_date_str, (date, datetime)):
            d = ipo_date_str if isinstance(ipo_date_str, date) else ipo_date_str.date()
        else:
            d = datetime.strptime(str(ipo_date_str)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    return (date.today() - d).days <= cutoff_days


def compute_scarcity_stats(
    peers: list[dict],
    *,
    target_ticker: str | None = None,
    liquidity_threshold_hkd: float = 5e7,  # 5000 万 HKD
    today: date | None = None,
    pipeline_companies_in_theme: list[str] | None = None,
) -> ScarcityStats:
    """从 peers 列表 (含市值 / 30 日均成交 / IPO 日期 / 首日涨幅) 算稀缺度统计.

    Args:
        peers: PeerSuggester + iFinD 拉到的同主题已上市 peer 列表
        target_ticker: 目标公司 ticker (从 peers 中排除, 如已混入)
        liquidity_threshold_hkd: 流动性枯竭阈值 (默认 5000 万 HKD/日)
        today: 用于计算"过去 12 月", 测试可注入
        pipeline_companies_in_theme: 同主题港股 IPO 排队中公司名列表
            (来自 ifind_ipo_queue + theme_classifier 命中). v2 流量稀缺度.

    Returns: ScarcityStats — 全部确定性, LLM 不参与
    """
    rationale: list[str] = []
    warnings: list[str] = []

    # 排除 target 自己
    peers = [
        p for p in (peers or [])
        if not target_ticker or (
            p.get("thscode") != target_ticker and p.get("ticker") != target_ticker
        )
    ]

    pipeline_companies = list(pipeline_companies_in_theme or [])
    pipeline_count = len(pipeline_companies)

    n = len(peers)
    if n == 0:
        warnings.append("peer 列表为空, 无法量化稀缺度")
        return ScarcityStats(
            listed_count_in_theme=0, market_cap_total_hkd_b=0.0,
            market_cap_top3_share=None, median_market_cap_hkd_b=None,
            liquidity_thinning_count=0, liquidity_thinning_ratio=None,
            avg_first_day_return_pct=None, recent_ipo_count_12m=0,
            pipeline_count_in_theme=pipeline_count,
            pipeline_companies=pipeline_companies,
            raw_scarcity_score=4.0,  # 没 peer 时偏稀缺, 但需 LLM 验证主题界定
            rationale=["peer 数为 0 → 表面独苗, 但需确认主题界定是否过窄"],
            warnings=warnings,
        )

    # 1) 市值分布
    caps = [c for c in (_compute_market_cap_hkd_b(p) for p in peers) if c is not None]
    cap_total = sum(caps) if caps else 0.0
    if caps:
        top3 = sorted(caps, reverse=True)[:3]
        top3_share = sum(top3) / cap_total if cap_total > 0 else None
        median_cap = statistics.median(caps)
    else:
        top3_share = None
        median_cap = None
        warnings.append("peer 市值数据缺失, 集中度未量化")

    # 2) 流动性枯竭计数
    thinning = 0
    has_liq_data = 0
    for p in peers:
        # 30 日均成交 (HKD)
        for key in ("avg_turnover_30d_hkd", "turnover_30d_hkd", "avg_turnover_30d"):
            v = _safe_float(p.get(key))
            if v is not None and v > 0:
                has_liq_data += 1
                if v < liquidity_threshold_hkd:
                    thinning += 1
                break
    thinning_ratio = (thinning / has_liq_data) if has_liq_data > 0 else None

    # 3) 最近 IPO 首日均值
    first_day_returns = [
        _safe_float(p.get("first_day_open_return")) for p in peers
        if p.get("first_day_open_return") is not None
    ]
    first_day_returns = [r for r in first_day_returns if r is not None]
    avg_fdr = statistics.mean(first_day_returns) if first_day_returns else None

    # 4) 过去 12 月 IPO 数量
    cutoff = today or date.today()
    recent_count = sum(
        1 for p in peers
        if _is_recent_ipo(p.get("ipo_date"))
    )

    # 5) 引擎初算 raw_scarcity_score (1-5, 5=最稀缺)
    # 启发式 (LLM 可在 rationale 里覆盖):
    # - 同主题已上市 ≤ 2 → +1.5 (独苗赛道)
    # - 3-5 → +1.0
    # - 6-10 → +0.5
    # - >10 → -0.5 (红海)
    # - 流动性枯竭 > 50% → -0.5 (即使少也不真稀缺)
    # - top3 市值占比 > 75% → -0.5 (龙头垄断, 中小被边缘化)
    # - 最近 12 月已发 ≥ 3 家 → -0.5 (边际稀缺度递减)
    score = 3.0  # 中性起点
    if n <= 2:
        score += 1.5
        rationale.append(f"同主题港股仅 {n} 家 → 独苗赛道, +1.5 分")
    elif n <= 5:
        score += 1.0
        rationale.append(f"同主题 {n} 家 → 偏稀缺, +1.0 分")
    elif n <= 10:
        score += 0.5
        rationale.append(f"同主题 {n} 家 → 中等供给, +0.5 分")
    else:
        score -= 0.5
        rationale.append(f"同主题 {n} 家 → 红海, -0.5 分")

    if thinning_ratio is not None and thinning_ratio > 0.5:
        score -= 0.5
        rationale.append(
            f"{thinning}/{has_liq_data} ({thinning_ratio:.0%}) 同主题公司 30 日均成交 "
            f"< 5000 万 HKD, 流动性枯竭 → -0.5"
        )

    if top3_share is not None and top3_share > 0.75:
        score -= 0.5
        rationale.append(f"Top 3 市值占比 {top3_share:.0%} > 75% 龙头垄断 → -0.5")

    if recent_count >= 3:
        score -= 0.5
        rationale.append(
            f"过去 12 月同主题已 IPO {recent_count} 家, 边际稀缺度递减 → -0.5"
        )

    # v2 流量稀缺度: 同主题排队中公司数
    # 4+ → -1.0 (供给将在 6-12 月内显著扩张, 边际稀缺度大幅压缩)
    # 2-3 → -0.5 (供给中度扩张)
    # 1 → -0.2 (轻微)
    # 0 → 0 (供给端无新增, 现有稀缺度持续)
    if pipeline_count >= 4:
        score -= 1.0
        rationale.append(
            f"同主题排队中 {pipeline_count} 家 → 6-12 月供给显著扩张, -1.0"
        )
    elif pipeline_count >= 2:
        score -= 0.5
        rationale.append(
            f"同主题排队中 {pipeline_count} 家 → 供给中度扩张, -0.5"
        )
    elif pipeline_count == 1:
        score -= 0.2
        rationale.append(f"同主题排队中 1 家 → 轻微供给扩张, -0.2")
    else:
        rationale.append("同主题无排队中公司, 现有稀缺度持续")

    score = max(1.0, min(5.0, score))

    return ScarcityStats(
        listed_count_in_theme=n,
        market_cap_total_hkd_b=round(cap_total, 2) if cap_total else 0.0,
        market_cap_top3_share=round(top3_share, 4) if top3_share is not None else None,
        median_market_cap_hkd_b=round(median_cap, 2) if median_cap is not None else None,
        liquidity_thinning_count=thinning,
        liquidity_thinning_ratio=round(thinning_ratio, 4) if thinning_ratio is not None else None,
        avg_first_day_return_pct=round(avg_fdr, 2) if avg_fdr is not None else None,
        recent_ipo_count_12m=recent_count,
        pipeline_count_in_theme=pipeline_count,
        pipeline_companies=pipeline_companies,
        raw_scarcity_score=round(score, 2),
        rationale=rationale,
        warnings=warnings,
    )


def render_stats_for_prompt(stats: ScarcityStats) -> str:
    """渲染稀缺度统计为 markdown, 喂给 ScarcityAgent prompt."""
    lines = [
        "**[引擎确定性稀缺度统计]** — 全部从 peer 数据自动算, LLM 在此基础上做定性论证。",
        "",
        f"- 同主题已上市公司数: **{stats.listed_count_in_theme}**",
        f"- 同主题总市值: {stats.market_cap_total_hkd_b} 亿 HKD",
    ]
    if stats.market_cap_top3_share is not None:
        lines.append(f"- Top 3 市值占比: {stats.market_cap_top3_share:.0%}")
    if stats.median_market_cap_hkd_b is not None:
        lines.append(f"- 中位市值: {stats.median_market_cap_hkd_b} 亿 HKD")
    if stats.liquidity_thinning_ratio is not None:
        lines.append(
            f"- 流动性枯竭家数 (30 日均成交 < 5000 万): "
            f"{stats.liquidity_thinning_count} ({stats.liquidity_thinning_ratio:.0%})"
        )
    if stats.avg_first_day_return_pct is not None:
        lines.append(f"- 最近同主题 IPO 首日开盘均值: {stats.avg_first_day_return_pct:+.2f}%")
    lines.append(f"- 过去 12 月同主题 IPO 数: {stats.recent_ipo_count_12m}")
    lines.append(
        f"- **同主题港股排队中 (申请版本/已通过聆讯): {stats.pipeline_count_in_theme} 家** "
        f"⭐ v2 流量稀缺度"
    )
    if stats.pipeline_companies:
        sample = stats.pipeline_companies[:5]
        more = "" if len(stats.pipeline_companies) <= 5 else f" 等 {len(stats.pipeline_companies)} 家"
        lines.append(f"  - 排队中: {', '.join(sample)}{more}")
    lines.append(f"- **引擎初算稀缺度: {stats.raw_scarcity_score} / 5**")
    if stats.rationale:
        lines.append("")
        lines.append("引擎打分依据:")
        for r in stats.rationale:
            lines.append(f"- {r}")
    if stats.warnings:
        lines.append("")
        lines.append("数据缺口:")
        for w in stats.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)
