"""一次性脚本: 重跑 DebateOrchestrator (Bull/Bear + Manager).

之前 debate_manager 输出空白根因:
- DECIDE tier 映射的 kimi-k2-thinking 把推理放在 reasoning_content, content 为空。
- client.py 已加 reasoning_content 回退, .env 已把 DECIDE 改 kimi-k2.6 (普通模型)。
- 现在重跑应该正常。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from src.agents.base import AgentContext
from src.agents.debate import DebateOrchestrator
from src.agents.extras import WorkflowExtras
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"
REPORTS_DIR = Path("reports") / PROJECT_ID
BRIEF_PAT = re.compile(r"^(\d{2})_(.+)\.brief\.md$")


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1

    extras = WorkflowExtras()
    ctx = AgentContext(
        project_id=PROJECT_ID,
        ticker=TICKER,
        company_name=COMPANY,
        industry=INDUSTRY,
        reports_dir=REPORTS_DIR,
        rag=None,
        extras=extras,
    )
    # 加载 1-6 步 brief 给 bull/bear 当 upstream
    loaded = []
    for f in sorted(REPORTS_DIR.iterdir()):
        m = BRIEF_PAT.match(f.name)
        if not m:
            continue
        step_no = int(m.group(1))
        agent_name = m.group(2)
        if step_no >= 7:
            continue
        ctx.briefs[agent_name] = f.read_text(encoding="utf-8")
        loaded.append((step_no, agent_name))
    logger.info(f"loaded {len(ctx.briefs)} upstream briefs: {loaded}")

    llm = LLMClient()
    debate = DebateOrchestrator(llm, max_rounds=2)
    report = debate.run(ctx)
    logger.info(f"manager_full len={len(report.full_report)}, brief len={len(report.brief)}")
    logger.info(f"brief 前 200: {report.brief[:200]}")

    debate._save_full_report(ctx, 7, report.full_report)
    (REPORTS_DIR / "07_debate_manager.brief.md").write_text(
        report.brief, encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
