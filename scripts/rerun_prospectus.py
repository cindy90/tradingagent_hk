"""一次性脚本: 重跑 ProspectusAnalystAgent (复用现有 RAG)。"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from src.agents.base import AgentContext
from src.agents.extras import WorkflowExtras
from src.agents.prospectus_analyst import ProspectusAnalystAgent
from src.agents.summarizer import Summarizer
from src.data.rag import ProspectusRAG
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"
REPORTS_DIR = Path("reports") / PROJECT_ID


def main() -> int:
    rag = ProspectusRAG(project_id=PROJECT_ID)
    if not rag.is_indexed():
        logger.error("RAG 未索引")
        return 1

    extras = WorkflowExtras()
    ctx = AgentContext(
        project_id=PROJECT_ID,
        ticker=TICKER,
        company_name=COMPANY,
        industry=INDUSTRY,
        reports_dir=REPORTS_DIR,
        rag=rag,
        extras=extras,
    )

    llm = LLMClient()
    agent = ProspectusAnalystAgent(llm, Summarizer(llm))
    report = agent.run(ctx)
    logger.info(f"prospectus 报告 {len(report.full_report)} 字, brief {len(report.brief)} 字")
    logger.info(f"brief 前 200: {report.brief[:200]}")

    agent._save_full_report(ctx, 1, report.full_report)
    (REPORTS_DIR / "01_prospectus_analyst.brief.md").write_text(
        report.brief, encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
