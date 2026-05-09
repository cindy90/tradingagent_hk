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
from src.data.prospectus import ProspectusLoader
from src.data.rag import ProspectusRAG
from src.data.ths_client import THSClient
from src.llm import LLMClient


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
        extras: dict | None = None,
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
                # 选择招股书最关键的 2-3 个章节作为缓存块（概要 + 风险因素）
                # 这里用启发式：取前 5 块文本作为静态缓存内容。
                head_blocks = [c.text for c in chunks[:3]]
                cached_blocks = ["\n\n".join(head_blocks)] if head_blocks else []
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
            extras=extras or {},
        )

    def run(
        self,
        *,
        ticker: str,
        company_name: str,
        industry: str,
        prospectus_pdf: str | Path | None = None,
        extras: dict | None = None,
    ) -> AgentContext:
        merged_extras = dict(extras or {})
        prefetched = self._prefetch_ths(ticker, industry)
        merged_extras.update(prefetched)

        ctx = self._build_ctx(ticker, company_name, industry, prospectus_pdf, merged_extras)
        logger.info(f"=== 工作流启动: {ctx.project_id} ===")

        for i, agent in enumerate(self.steps, start=1):
            logger.info(f"--- 步骤 {i}/{len(self.steps)}: {agent.name} ---")
            try:
                report = agent.run(ctx)
                agent._save_full_report(ctx, i, report.full_report)
                # 简报也单独落盘，便于审查
                brief_path = ctx.reports_dir / f"{i:02d}_{agent.name}.brief.md"
                brief_path.write_text(report.brief, encoding="utf-8")
            except Exception as e:
                logger.exception(f"Agent {agent.name} 失败: {e}")
                err_path = ctx.reports_dir / f"{i:02d}_{agent.name}.ERROR.md"
                err_path.write_text(f"# {agent.name} 执行失败\n\n```\n{e}\n```\n", encoding="utf-8")

        # token 使用账本
        (ctx.reports_dir / "_token_usage.md").write_text(
            "# Token 使用账本\n\n" + self.llm.ledger.summary(), encoding="utf-8"
        )
        logger.info(f"=== 工作流完成，报告目录: {ctx.reports_dir} ===")
        return ctx
