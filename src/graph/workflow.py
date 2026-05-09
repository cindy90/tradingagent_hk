"""基石投资分析主工作流。

执行顺序（每步落盘 markdown 报告，下游 Agent 只读 brief 摘要）：
  1) 招股书深度分析     → prospectus_analyst
  2) 行业研究           → industry
  3) 宏观策略           → macro
  4) 可比公司估值       → comparable
  5) 技术发展趋势       → tech_trend
  6) 二级市场情绪       → sentiment
  7) Bull/Bear 辩论     → debate_manager
  8) 风控独立评估       → risk
  9) 最终投决           → decision

未来可改造为 LangGraph，当前用线性 + 显式步骤，便于调试和理解。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import get_settings
from src.agents.base import AgentContext, BaseAgent
from src.agents.comparable import ComparableAgent
from src.agents.debate import DebateOrchestrator
from src.agents.decision import DecisionAgent
from src.agents.industry import IndustryAgent
from src.agents.macro import MacroAgent
from src.agents.prospectus_analyst import ProspectusAnalystAgent
from src.agents.risk import RiskAgent
from src.agents.sentiment import SentimentAgent
from src.agents.summarizer import Summarizer
from src.agents.tech_trend import TechTrendAgent
from src.agents.extras import WorkflowExtras
from src.data.prospectus import ProspectusChunk, ProspectusLoader
from src.data.rag import ProspectusRAG
from src.data.ths_client import THSClient
from src.llm import LLMClient

# 高信息密度章节关键词（按优先级排序）。命中后作为 cached_system_blocks 内容，
# 跨 Agent 复用 prompt cache，且每个 Agent 都白拿这些核心上下文。
KEY_CACHED_SECTIONS = [
    "概要",       # 招股书第一章，浓缩全文
    "风险因素",
    "业务",
    "募集资金用途",
    "募资",
    "财务资料",
    "管理层讨论",
    "RISK FACTORS",
    "BUSINESS",
    "USE OF PROCEEDS",
]


def select_cached_blocks(
    chunks: list[ProspectusChunk],
    *,
    max_blocks: int = 4,
    max_chars_per_block: int = 6000,
    max_total_chars: int = 20000,
) -> list[str]:
    """从切块中挑选关键章节文本作为 prompt cache 内容。

    匹配规则:
      1) chunk.section 包含 KEY_CACHED_SECTIONS 关键词
      2) 同一 section 只取第一块（避免重复）
      3) 总字符数 < max_total_chars
      4) 没有命中关键词时退化为前 3 块（保留原行为）
    """
    selected: list[str] = []
    seen_sections: set[str] = set()
    total = 0

    for kw in KEY_CACHED_SECTIONS:
        for c in chunks:
            sec = (c.section or "").strip()
            if not sec or sec in seen_sections:
                continue
            if kw not in sec:
                continue
            text = c.text[:max_chars_per_block]
            if total + len(text) > max_total_chars:
                break
            selected.append(text)
            seen_sections.add(sec)
            total += len(text)
            break  # 每个关键词只取一块
        if len(selected) >= max_blocks or total >= max_total_chars:
            break

    if not selected:
        # fallback: 前 3 块（保留旧行为，避免完全没 cache）
        selected = [c.text[:max_chars_per_block] for c in chunks[:3]]

    return selected


class CornerstoneWorkflow:
    def __init__(self, llm: LLMClient | None = None, debate_max_rounds: int | None = None):
        self.llm = llm or LLMClient()
        self.summarizer = Summarizer(self.llm)
        s = get_settings()
        max_rounds = debate_max_rounds if debate_max_rounds is not None else s.debate_max_rounds

        self.steps: list[BaseAgent] = [
            ProspectusAnalystAgent(self.llm, self.summarizer),
            IndustryAgent(self.llm, self.summarizer),
            MacroAgent(self.llm, self.summarizer),
            ComparableAgent(self.llm, self.summarizer),
            TechTrendAgent(self.llm, self.summarizer),
            SentimentAgent(self.llm, self.summarizer),
            DebateOrchestrator(self.llm, max_rounds=max_rounds),
            RiskAgent(self.llm, self.summarizer),
            DecisionAgent(self.llm),
        ]

    @staticmethod
    def _prefetch_ths(ticker: str, industry: str) -> dict:
        """跑 Agent 前先把 THS_BD/THS_EDB/THS_DR 数据预取一遍，丢到 ctx.extras。

        失败时返回空字典对应键，下游 Agent 自行处理空值。
        """
        ths = THSClient()
        if not ths.configured:
            logger.info("THS 未配置，跳过基础数据/宏观/研报预取")
            return {}

        code = ticker if "." in ticker else f"{ticker.zfill(5)}.HK"
        out: dict[str, Any] = {}

        try:
            bd = ths.basic_data(codes=code)
            out["company_basic"] = bd.get(code, {}) if isinstance(bd, dict) else {}
            logger.info(f"[Prefetch] basic_data 字段数: {len(out['company_basic'])}")
        except Exception as e:
            logger.warning(f"[Prefetch] basic_data 失败: {e}")
            out["company_basic"] = {}

        try:
            macro = ths.edb()
            out["macro_indicators"] = macro
            logger.info(f"[Prefetch] EDB 指标数: {len(macro)}")
        except Exception as e:
            logger.warning(f"[Prefetch] edb 失败: {e}")
            out["macro_indicators"] = {}

        try:
            reports = ths.research_reports(codes=code, industry=industry)
            out["industry_research"] = reports[:20]  # 限制条数控制 token
            logger.info(f"[Prefetch] 研报条数: {len(out['industry_research'])}")
        except Exception as e:
            logger.warning(f"[Prefetch] research_reports 失败: {e}")
            out["industry_research"] = []

        return out

    @staticmethod
    def _build_ctx(
        ticker: str,
        company_name: str,
        industry: str,
        prospectus_pdf: str | Path | None,
        extras: WorkflowExtras | None = None,
    ) -> AgentContext:
        s = get_settings()
        project_id = f"{ticker}_{datetime.now():%Y%m%d_%H%M%S}"
        reports_dir = s.reports_dir / project_id
        reports_dir.mkdir(parents=True, exist_ok=True)

        rag: ProspectusRAG | None = None
        cached_blocks: list[str] = []
        if prospectus_pdf:
            pdf_path = Path(prospectus_pdf)
            if pdf_path.exists():
                logger.info(f"加载招股书 {pdf_path}")
                chunks = ProspectusLoader(pdf_path).load_chunks()
                rag = ProspectusRAG(project_id=project_id)
                rag.index(chunks)
                # 按章节关键词挑选高信息密度章节作为 prompt cache 内容
                selected = select_cached_blocks(chunks)
                cached_blocks = ["\n\n---\n\n".join(selected)] if selected else []
                logger.info(
                    f"cached_blocks: {len(selected)} 个关键章节 / "
                    f"{sum(len(s) for s in selected)} 字符"
                )
            else:
                logger.warning(f"招股书 PDF 不存在: {pdf_path}")

        return AgentContext(
            project_id=project_id,
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            reports_dir=reports_dir,
            rag=rag,
            cached_blocks=cached_blocks,
            extras=extras or WorkflowExtras(),
        )

    def run(
        self,
        *,
        ticker: str,
        company_name: str,
        industry: str,
        prospectus_pdf: str | Path | None = None,
        extras: dict | None = None,
        use_case_rag: bool = True,
    ) -> AgentContext:
        prefetched = self._prefetch_ths(ticker, industry)
        wf_extras = WorkflowExtras.from_dict({**(extras or {}), **prefetched})

        ctx = self._build_ctx(ticker, company_name, industry, prospectus_pdf, wf_extras)

        # 闭环关键 (Phase C): 检索历史相似案例，注入到 cached_blocks
        if use_case_rag:
            try:
                self._inject_similar_cases(ctx)
            except Exception as e:
                logger.warning(f"CaseRAG 检索失败（不影响主流程）: {e}")

        logger.info(f"=== 工作流启动: {ctx.project_id} ===")

        for i, agent in enumerate(self.steps, start=1):
            fatal_tag = " (fatal)" if agent.fatal else ""
            logger.info(f"--- 步骤 {i}/{len(self.steps)}: {agent.name}{fatal_tag} ---")
            try:
                report = agent.run(ctx)
                agent._save_full_report(ctx, i, report.full_report)
                brief_path = ctx.reports_dir / f"{i:02d}_{agent.name}.brief.md"
                brief_path.write_text(report.brief, encoding="utf-8")
            except Exception as e:
                logger.exception(f"Agent {agent.name} 失败: {e}")
                err_path = ctx.reports_dir / f"{i:02d}_{agent.name}.ERROR.md"
                err_path.write_text(
                    f"# {agent.name} 执行失败\n\n```\n{type(e).__name__}: {e}\n```\n",
                    encoding="utf-8",
                )
                if agent.fatal:
                    # 写出 token 账本后再抛，至少留下成本痕迹
                    self._write_ledger(ctx)
                    logger.error(f"致命 Agent [{agent.name}] 失败，工作流中止")
                    raise
                # 非致命：注入失败标记到 brief，下游 LLM 能明确感知信息缺失
                ctx.briefs[agent.name] = (
                    f"**[执行失败]** 本环节因 `{type(e).__name__}` 异常未产出有效内容。\n"
                    f"信息缺失影响范围：{agent.description or agent.name}。\n"
                    f"请在你的分析与决策中**显式承认这部分信息空白**，"
                    f"不要凭空推断或假装拥有这部分数据。"
                )

        self._write_ledger(ctx)
        # 自动落库 Prediction（Phase A）
        try:
            self._persist_prediction(ctx)
        except Exception as e:
            logger.warning(f"prediction 落库失败（不影响报告产出）: {e}")
        logger.info(f"=== 工作流完成，报告目录: {ctx.reports_dir} ===")
        return ctx

    @staticmethod
    def _inject_similar_cases(ctx: AgentContext) -> None:
        """从 CaseRAG 检索相似历史案例，把 prompt 文本拼到 ctx.cached_blocks 末尾。

        策略：找 top-3 相似案例（行业 + 估值规模 + 已有 outcome），
        渲染成 markdown 块，作为新的 cached_block 追加。
        所有 Agent 通过 cached_system_blocks 都能看到，但 Decision 受益最大。
        """
        from src.feedback import CaseRAG
        rag = CaseRAG()
        if rag.count() == 0:
            logger.info("CaseRAG 当前无历史案例，跳过注入")
            return
        # 估值规模未知（IPO 之前没决议），先按行业匹配
        valuation_hint = None
        if ctx.extras.peers:
            # 用同行平均规模作为粗略 hint
            mids = [
                p.get("market_cap_hkd_b")
                for p in ctx.extras.peers
                if isinstance(p, dict) and p.get("market_cap_hkd_b")
            ]
            if mids:
                valuation_hint = sum(mids) / len(mids)
        hits = rag.search(industry=ctx.industry, valuation_mid=valuation_hint, k=3)
        if not hits:
            return
        prompt_block = CaseRAG.format_for_prompt(hits)
        if prompt_block:
            ctx.cached_blocks.append(prompt_block)
            logger.info(f"注入 {len(hits)} 个历史相似案例到 cached_blocks")
            # 标记进 Prediction.cogalpha_features_used (后续 _persist_prediction 用)
            features = ctx.extras.misc.setdefault("cogalpha_features", set())
            if isinstance(features, set):
                features.add("case_rag")

    def _write_ledger(self, ctx: AgentContext) -> None:
        from src.llm.pricing import estimate_total_cost_cny
        from src.llm.router import ModelTier, resolve_model
        tier_to_model = {t.value: resolve_model(t) for t in ModelTier}
        cost = estimate_total_cost_cny(self.llm.ledger.by_tier, tier_to_model)
        (ctx.reports_dir / "_token_usage.md").write_text(
            "# Token 使用账本\n\n"
            + self.llm.ledger.summary()
            + f"\n\n**预估成本: ¥{cost}**\n\n"
            + "Tier → Model:\n"
            + "\n".join(f"- {t}: `{m}`" for t, m in tier_to_model.items()),
            encoding="utf-8",
        )

    def _persist_prediction(self, ctx: AgentContext) -> None:
        """把决议结果落到 feedback DB。决议缺失时跳过。"""
        from datetime import datetime as _dt

        from src.feedback import FeedbackStore
        from src.feedback.models import Prediction
        from src.llm.pricing import estimate_total_cost_cny
        from src.llm.router import ModelTier, resolve_model

        decision = ctx.extras.decision_json
        if not decision:
            logger.info("决议 JSON 缺失，跳过 prediction 落库")
            return

        amount = decision.get("suggested_amount_usd_million") or [0, 0]
        valuation = decision.get("valuation_range_hkd_billion") or {}

        agg = self.llm.ledger.by_tier
        total_in = sum(s.get("input", 0) for s in agg.values())
        total_out = sum(s.get("output", 0) for s in agg.values())
        total_cache = sum(s.get("cache_read", 0) for s in agg.values())
        tier_to_model = {t.value: resolve_model(t) for t in ModelTier}

        from config import get_settings
        s = get_settings()

        score_cards = ctx.extras.misc.get("score_cards", {})
        features_used = ["scoring_card"]
        cog_features = ctx.extras.misc.get("cogalpha_features")
        if cog_features:
            features_used.extend(sorted(cog_features))

        p = Prediction(
            project_id=ctx.project_id,
            ticker=ctx.ticker,
            company_name=ctx.company_name,
            industry=ctx.industry,
            decision_date=_dt.now(),
            recommendation=str(decision.get("recommendation", "")),
            confidence=str(decision.get("confidence", "")),
            valuation_low=valuation.get("low"),
            valuation_mid=float(valuation.get("mid", 0) or 0),
            valuation_high=valuation.get("high"),
            anchor_method=str(valuation.get("anchor_method", "")),
            anchor_logic=str(valuation.get("anchor_logic", "")),
            ipo_pricing_view=str(decision.get("ipo_pricing_view", "")),
            suggested_amount_low_usd_m=float(amount[0]) if len(amount) > 0 else 0,
            suggested_amount_high_usd_m=float(amount[1]) if len(amount) > 1 else 0,
            key_supports=list(decision.get("key_supports", [])),
            key_risks=list(decision.get("key_risks", [])),
            deal_conditions=list(decision.get("deal_conditions", [])),
            monitoring_kpis=list(decision.get("monitoring_kpis", [])),
            agent_score_cards=score_cards,
            model_provider=s.llm_provider,
            model_tier_models=tier_to_model,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cache_read_tokens=total_cache,
            estimated_cost_cny=estimate_total_cost_cny(agg, tier_to_model),
            cogalpha_features_used=features_used,
            reports_dir_path=str(ctx.reports_dir),
            status="open",
        )
        store = FeedbackStore()
        try:
            pid = store.save_prediction(p)
            logger.info(f"prediction 已落库 id={pid}")
        finally:
            store.close()
