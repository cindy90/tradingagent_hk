"""一次性脚本: 用 PeerSuggester 选出的 peers 重跑 industry agent."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.agents.base import AgentContext
from src.agents.extras import WorkflowExtras
from src.agents.industry import IndustryAgent
from src.agents.summarizer import Summarizer
from src.data.rag import ProspectusRAG
from src.graph.workflow import _prefetch_ifind_sdk
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"
DEFAULT_PEERS = ["02432", "01021"]
REPORTS_DIR = Path("reports") / PROJECT_ID


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1
    peers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PEERS
    # 用 H2254 拿珞石真实数据（02670 在 iFinD 是云迹）
    sdk_data = _prefetch_ifind_sdk(
        peers, target_ticker="H2254", expected_target_name=COMPANY
    )

    rag = ProspectusRAG(project_id=PROJECT_ID)
    if not rag.is_indexed():
        logger.error("RAG 未索引")
        return 1

    extras = WorkflowExtras()
    extras.peers = sdk_data["peers"]
    extras.target_valuation = sdk_data.get("target_valuation")

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
    agent = IndustryAgent(llm, Summarizer(llm))
    report = agent.run(ctx)
    logger.info(f"industry brief 前 200 字: {report.brief[:200]}")
    agent._save_full_report(ctx, 2, report.full_report)
    (REPORTS_DIR / "02_industry.brief.md").write_text(report.brief, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
