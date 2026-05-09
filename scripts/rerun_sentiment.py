"""一次性脚本: 用 iFinD SDK 拿真实 peer 行情, 重跑 sentiment agent.

验证:
1. SDK 返回的 peer K 线 + IPO 信息正确填到 ctx.extras
2. 新版 sentiment.py prompt 把数据渲染成 markdown 表喂 LLM
3. LLM 引用真实股价 / 上市日期 / 招股价, 不再编 "2024 年 12 月" 之类幻觉
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.agents.base import AgentContext
from src.agents.extras import WorkflowExtras
from src.agents.sentiment import SentimentAgent
from src.agents.summarizer import Summarizer
from src.data.ths_client import THSClient
from src.graph.workflow import _prefetch_ifind_sdk
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"

# 默认对标: 越疆(港股协作机器人直接竞品) / 优必选(港股人形机器人) / 地平线机器人(港股 AI 芯片)
DEFAULT_PEERS = ["02432", "09880", "09660"]

REPORTS_DIR = Path("reports") / PROJECT_ID


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1

    # 用 SDK 拉真实 peer 数据
    peers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PEERS
    logger.info(f"peers={peers}")
    sdk_data = _prefetch_ifind_sdk(peers)
    logger.info(
        f"SDK 拉到 peers={len(sdk_data['peers'])} 条 IPO 信息, "
        f"{len(sdk_data['peer_recent_quotes'])} 条 K 线"
    )

    extras = WorkflowExtras()
    extras.peers = sdk_data["peers"]
    extras.peer_recent_quotes = sdk_data["peer_recent_quotes"]
    extras.recent_hk_ipos = sdk_data["recent_hk_ipos"]
    # 注入 EDB 宏观数据 (含南向资金 series), sentiment 第一节用
    ths = THSClient()
    if ths.configured:
        extras.macro_indicators = ths.edb()
        logger.info(f"EDB 宏观: {list(extras.macro_indicators.keys())}")

    ctx = AgentContext(
        project_id=PROJECT_ID,
        ticker=TICKER,
        company_name=COMPANY,
        industry=INDUSTRY,
        reports_dir=REPORTS_DIR,
        rag=None,
        extras=extras,
    )

    llm = LLMClient()
    agent = SentimentAgent(llm, Summarizer(llm))
    report = agent.run(ctx)
    logger.info(f"sentiment brief (前 200 字): {report.brief[:200]}")

    agent._save_full_report(ctx, 6, report.full_report)
    (REPORTS_DIR / "06_sentiment.brief.md").write_text(
        report.brief, encoding="utf-8"
    )
    logger.info(f"已重写 {REPORTS_DIR}/06_sentiment.md 和 06_sentiment.brief.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
