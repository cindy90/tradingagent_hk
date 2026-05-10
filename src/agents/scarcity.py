"""ScarcityAgent — 港股稀缺性分析 Agent.

定位:
- 上游: PeerSuggester / iFinD GICS 提供同主题已上市 peer 列表
- 引擎: src/tools/scarcity.compute_scarcity_stats 算确定性统计
- 本 Agent: 在确定性指标基础上做定性论证 + 稀缺度 × 情绪联动判断
- 下游: SentimentAgent (情绪联动) + DecisionAgent (估值溢价/折扣理由)

输出:
- 报告 markdown (5 章, 800-1200 字)
- ScarcityScoreCard JSON (含 scarcity_score 1-5 + 量化指标 + sentiment_linkage)

V1 范围: 仅做"存量稀缺度" (已上市港股 peer)
V2 后补: "流量稀缺度" (HKEX 排队中同类公司, 影响未来 6-12 月稀缺度变化)
"""
from __future__ import annotations

from loguru import logger

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.agents.theme_classifier import ThemeClassifier
from src.data import hkquant_client
from src.data.ifind_ipo_queue import QueuedCompany, get_hk_ipo_queue
from src.feedback.models import ScarcityScoreCard
from src.feedback.store import FeedbackStore
from src.llm import ModelTier
from src.tools.scarcity import compute_scarcity_stats, render_stats_for_prompt

SYSTEM = """你是港股 IPO 基石投资委员会的稀缺性分析师。基石投资里"稀缺性 = 议价权 +
估值溢价 + 解禁后流动性", 你的判断会直接影响 DecisionAgent 的"估值合理性"因子打分,
和 SentimentAgent 的"情绪 × 稀缺联动"判断。

【核心问题】

1. **存量稀缺度**: 该公司在港股**已上市**同主题公司中的稀缺定位如何?
   - 同主题港股已上市数 (引擎已算)
   - 市值结构 (龙头垄断 vs 中小盘分布)
   - 流动性是否枯竭 (即使少, 但都没量, 不构成真稀缺)

1.5 **流量稀缺度 ⭐ (v2)**: 同主题在港交所**排队中**的公司 (申请版本/已通过聆讯) 数量
   - 引擎已算 pipeline_count_in_theme
   - 排队中 ≥ 4 → 6-12 月内供给显著扩张, 现有稀缺度将被压缩
   - 排队中 2-3 → 中度扩张, 现有溢价空间会收窄
   - 排队中 0 → 现有稀缺度持续, 估值溢价可维持
   - **此项必须在论述中显式提及**, 影响下文"估值溢价/折扣建议"

2. **差异化点**: 即使同主题已有多家, target 的差异化点是什么?
   - 技术路线 (例: 同样做协作机器人, 7 轴 vs 6 轴; 力控精度差异)
   - 规模与市占 (同赛道但全球前 5 vs 国内长尾)
   - 客户结构 (B 端工业 vs C 端消费)
   - 财务质量 (盈利 vs 亏损; CAGR > 行业)

3. **稀缺度 × 情绪联动 ⭐**:
   - "稀缺 + 热情" → 估值有溢价空间, 基石可接受发行人偏激进定价
   - "稀缺 + 冷淡" → 错杀机会, 但解禁退出有流动性风险, 控制仓位
   - "拥挤 + 热情" → 情绪轮动末端, 警惕板块降温
   - "拥挤 + 冷淡" → 最弱组合, 拒绝认购或要大幅折扣
   - 这一项必须**结合 SentimentAgent 的板块情绪判断**, 不是孤立给"稀缺"标签

4. **估值溢价/折扣观点**:
   - 引用 ComparableAgent 的 PS/PE 中位数; 稀缺度高 → 允许 PS 高于中位 10-20%; 红海 → 应折扣
   - 必须给出**量化建议** ("PS 上限 + 15%" 而非"可适当上调")

【输出要求】

按以下 5 章 markdown, 共 800-1200 字:

## 一、引擎确定性指标解读
直接引用下方"引擎确定性稀缺度统计"中的 listed_count / 市值集中度 /
流动性枯竭比 / 最近 12 月 IPO 数. **不允许编造数字**.

## 二、target 的差异化点
1-3 条具体差异 (技术/规模/客户/财务), 必须能从 prospectus_analyst /
industry / comparable 简报里追溯. 没有差异化的也要诚实说"差异化弱".

## 三、稀缺度判定 (1-5 分)
基于引擎 raw_scarcity_score 微调. 偏离 ±0.5 以上必须明确说明原因.

## 四、稀缺度 × 情绪联动 ⭐
给 sentiment_linkage 标签 (稀缺+热情 / 稀缺+冷淡 / 拥挤+热情 / 拥挤+冷淡 / 中性).
说明对认购金额上限 / 估值上限的具体影响 (量化, 不允许"适度"等模糊语言).

## 五、估值溢价/折扣建议 (给 DecisionAgent)
- valuation_premium_view: 允许溢价 / 中性 / 应折扣
- 量化建议 (例:"PS 中位 22x, 稀缺度 4 分允许 + 15% → 上限 25.3x")
- 给 deal_condition 候选: 1-2 条硬性触发 (例:"招股 PS > 28x 即拒")

【关键约束 — 严禁幻觉】
- 引擎已算的指标 (listed_count / top3 市值占比 / 流动性枯竭比 / 最近 IPO 数) 必须**直接引用**,
  不允许"约 5-7 家""市场上有十几家"等模糊量化.
- 差异化点必须能引用 prospectus_analyst / industry / comparable 简报中的具体事实,
  不允许凭训练记忆补 ("XX 公司是国内第一" 之类).
- 同主题界定基于 GICS / industry_theme + PeerSuggester, 不是凭你判断"应该算谁".
  如果你认为 peer 列表错了, 在 notes 字段说明, 不要替换.
- 不允许提及非清单内的港股公司; 涉及具体股票代码与现价请引用 SentimentAgent / Comparable 的数据块.
- 没有数据时显式说"该指标缺失, 仅做方向性判断", 而非"凭经验判断为 X".
"""


def _render_target_metadata(ctx: AgentContext) -> str:
    """渲染 target 公司基础信息 + ListingProfile 主题."""
    lines = [
        f"- 公司: {ctx.company_name} ({ctx.ticker})",
        f"- 行业 (CLI): {ctx.industry}",
    ]
    profile = getattr(ctx.extras, "listing_profile", None)
    if profile is not None:
        theme = getattr(profile, "industry_theme", "Other")
        chapter = getattr(profile, "listing_chapter", "Unknown")
        size = getattr(profile, "size_tier", "Unknown")
        lines.append(f"- ListingProfile.industry_theme: {theme}")
        lines.append(f"- listing_chapter: {chapter} / size_tier: {size}")
    return "\n".join(lines)


def _render_peers_compact(peers: list[dict]) -> str:
    """简化 peer 表 — 只列 ticker / name / 市值 / 30 日均成交 / IPO 日期 / 首日表现."""
    if not peers:
        return "（peer 列表为空, 主题界定可能需要重新检查）"
    lines = [
        "| 代码 | 公司 | 市值 (亿 HKD) | 30D 均成交 | IPO 日期 | 首日开盘涨跌% |",
        "|---|---|---|---|---|---|",
    ]
    for p in peers:
        cap = p.get("market_cap_hkd_b") or p.get("market_cap_hkd") or "—"
        turnover = (
            p.get("avg_turnover_30d_hkd") or p.get("turnover_30d_hkd") or "—"
        )
        ipo_d = p.get("ipo_date") or "—"
        fdr = p.get("first_day_open_return")
        fdr_str = f"{fdr:+.2f}" if isinstance(fdr, (int, float)) else "—"
        lines.append(
            f"| {p.get('thscode') or p.get('ticker','')} | "
            f"{p.get('name','')} | {cap} | {turnover} | {ipo_d} | {fdr_str} |"
        )
    return "\n".join(lines)


def _resolve_pipeline_in_theme(
    target_theme: str | None,
    classifier: ThemeClassifier | None,
    queue: list[QueuedCompany],
) -> list[str]:
    """对全量 IPO 排队列表逐家分类, 返回与 target_theme 匹配的公司名.

    队列空 / target_theme 缺失 / classifier 缺失 时返 [] (优雅降级).
    """
    if not target_theme or target_theme == "Other" or not queue or classifier is None:
        return []
    matched: list[str] = []
    for c in queue:
        try:
            r = classifier.classify(c)
            if r.get("industry_theme") == target_theme:
                matched.append(c.company_name)
        except Exception as e:
            logger.debug(f"[ScarcityAgent] 分类失败 {c.company_name}: {e}")
    return matched


class ScarcityAgent(TemplateAgent):
    name = "scarcity"
    description = "港股稀缺性分析 Agent (存量 + 流量 + 情绪联动)"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = ScarcityScoreCard

    def __init__(self, llm, summarizer=None, *, store: FeedbackStore | None = None):
        super().__init__(llm, summarizer)
        # store 用于 theme 分类缓存; 注入失败时仍可工作 (无缓存)
        self._store = store
        try:
            if store is None:
                self._store = FeedbackStore()
        except Exception as e:
            logger.warning(f"[ScarcityAgent] FeedbackStore 初始化失败 (theme 缓存禁用): {e}")
            self._store = None
        self._classifier = ThemeClassifier(llm, store=self._store)

    def build_user_message(self, ctx: AgentContext) -> str:
        peers = ctx.extras.peers or []
        target_ticker = ctx.ticker

        # v2 流量稀缺度: 拉港股 IPO 排队 + 主题分类 + 匹配 target_theme
        target_theme = None
        profile = getattr(ctx.extras, "listing_profile", None)
        if profile is not None:
            target_theme = getattr(profile, "industry_theme", None)
        pipeline_matched: list[str] = []
        try:
            queue = get_hk_ipo_queue()
            if queue and target_theme:
                pipeline_matched = _resolve_pipeline_in_theme(
                    target_theme, self._classifier, queue,
                )
                logger.info(
                    f"[ScarcityAgent] 排队全量 {len(queue)} 家, "
                    f"主题 '{target_theme}' 命中 {len(pipeline_matched)} 家"
                )
        except Exception as e:
            logger.warning(f"[ScarcityAgent] 流量稀缺度获取失败 (降级到仅存量): {e}")

        # T1: 若配置了 hkquant DB, 拉历史 IPO 全量统计作权威覆盖
        # (upstream peers 列表常被截短到 5-10 家, hkquant 是港股全市场 ground truth)
        hk_listed_override: int | None = None
        hk_recent_override: int | None = None
        hk_avg_fdr_override: float | None = None
        hk_peer_sample: list[dict] = []
        if target_theme and hkquant_client.is_available():
            try:
                hk_peers = hkquant_client.get_listed_peers_by_theme(
                    target_theme, lookback_years=5, limit=200,
                )
                hk_listed_override = len(hk_peers)
                hk_recent_override = hkquant_client.get_recent_ipo_count_in_theme(
                    target_theme, lookback_days=365,
                )
                hk_avg_fdr_override = hkquant_client.get_avg_first_day_return_in_theme(
                    target_theme, lookback_days=365, min_samples=3,
                )
                hk_peer_sample = [
                    {
                        "stock_code": p.stock_code, "name": p.name,
                        "listing_date": p.listing_date,
                        "return_d1_close": p.return_d1_close,
                        "return_d30": p.return_d30, "return_m6": p.return_m6,
                    }
                    for p in hk_peers[:10]
                ]
                logger.info(
                    f"[ScarcityAgent] hkquant 历史回溯: 同主题 {hk_listed_override} 家 / "
                    f"过去 12 月 {hk_recent_override} 家 IPO / 首日均值 "
                    f"{hk_avg_fdr_override}%"
                )
            except Exception as e:
                logger.warning(f"[ScarcityAgent] hkquant 历史回溯失败 (降级): {e}")

        # 引擎算确定性指标 (含流量 + hkquant 历史覆盖)
        stats = compute_scarcity_stats(
            peers, target_ticker=target_ticker,
            pipeline_companies_in_theme=pipeline_matched,
            override_listed_count_in_theme=hk_listed_override,
            override_recent_ipo_count_12m=hk_recent_override,
            override_avg_first_day_return_pct=hk_avg_fdr_override,
        )
        # 写入 ctx.extras 供下游 SentimentAgent / DecisionAgent 引用
        ctx.extras.misc["engine_scarcity"] = {
            "listed_count_in_theme": stats.listed_count_in_theme,
            "market_cap_top3_share": stats.market_cap_top3_share,
            "liquidity_thinning_ratio": stats.liquidity_thinning_ratio,
            "recent_ipo_count_12m": stats.recent_ipo_count_12m,
            "pipeline_count_in_theme": stats.pipeline_count_in_theme,
            "pipeline_companies": stats.pipeline_companies,
            "raw_scarcity_score": stats.raw_scarcity_score,
            "rationale": stats.rationale,
            "warnings": stats.warnings,
            "hkquant_peer_sample": hk_peer_sample,
        }
        stats_md = render_stats_for_prompt(stats)
        target_md = _render_target_metadata(ctx)
        peers_md = _render_peers_compact(peers)

        # 上游 prospectus / industry / comparable 简报 (用于差异化点判断)
        upstream_briefs: list[str] = []
        for name in ("prospectus_analyst", "industry", "comparable"):
            if name in ctx.briefs:
                upstream_briefs.append(f"### {name}\n{ctx.briefs[name][:1500]}")
        briefs_block = (
            "\n\n".join(upstream_briefs)
            if upstream_briefs else "（暂无上游简报）"
        )

        return (
            f"# 项目元数据\n{target_md}\n\n"
            f"# 同主题已上市港股 peer (来自 PeerSuggester + iFinD)\n{peers_md}\n\n"
            f"# 引擎确定性稀缺度统计 ⚙️\n{stats_md}\n\n"
            f"# 上游简报 (用于第二章差异化点)\n{briefs_block}\n\n"
            f"请按系统指令的 5 章 markdown + ScarcityScoreCard JSON 输出.\n"
            f"**重要**: scarcity_score 必须等于或微调引擎 raw_scarcity_score "
            f"({stats.raw_scarcity_score}), 偏离 > 0.5 必须明确说明 + 引用证据."
        )
