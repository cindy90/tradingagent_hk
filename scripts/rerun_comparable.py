"""一次性脚本: 用 PeerSuggester 提取的真实 peers (越疆+华沿) 重跑 comparable agent."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.agents.base import AgentContext
from src.agents.comparable import ComparableAgent
from src.agents.extras import WorkflowExtras
from src.agents.summarizer import Summarizer
from src.data.rag import ProspectusRAG
from src.graph.workflow import _prefetch_ifind_sdk
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"

# PeerSuggester 验证选出的 2 个真实可比
DEFAULT_PEERS = ["02432", "01021"]

REPORTS_DIR = Path("reports") / PROJECT_ID


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1

    peers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PEERS
    logger.info(f"peers={peers}")

    # 注: TICKER=02670 在 iFinD 实际是云迹, 必须用 H2254 副牌代码取珞石
    sdk_data = _prefetch_ifind_sdk(
        peers, target_ticker="H2254", expected_target_name=COMPANY
    )
    extras = WorkflowExtras()
    extras.peers = sdk_data["peers"]
    extras.peer_recent_quotes = sdk_data["peer_recent_quotes"]
    extras.target_valuation = sdk_data.get("target_valuation")
    tv = sdk_data.get("target_valuation") or {}
    if tv.get("revenue"):
        extras.target_revenue = float(tv["revenue"])
    if tv.get("net_profit"):
        extras.target_net_profit = float(tv["net_profit"])

    rag = ProspectusRAG(project_id=PROJECT_ID)
    if not rag.is_indexed():
        logger.warning("RAG 未索引, 第四节招股价区间无法引用招股书")
        rag = None

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
    agent = ComparableAgent(llm, Summarizer(llm))
    report = agent.run(ctx)
    logger.info(f"comparable brief 前 200 字: {report.brief[:200]}")

    agent._save_full_report(ctx, 4, report.full_report)
    (REPORTS_DIR / "04_comparable.brief.md").write_text(
        report.brief, encoding="utf-8"
    )
    logger.info(f"已重写 {REPORTS_DIR}/04_comparable.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
