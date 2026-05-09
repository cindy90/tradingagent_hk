"""一次性脚本: 复用已生成的 8 个 brief, 只重跑 DecisionAgent + 生成 FINAL_MEMO.

用途: 02670 首跑时 decision 步骤 504 超时, 前 8 个 agent 输出已落盘.
此脚本避免再花 token/时间重跑前 8 步, 仅重新生成决议+终稿.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from loguru import logger

from src.agents.base import AgentContext
from src.agents.decision import DecisionAgent
from src.graph.workflow import persist_prediction
from src.llm import LLMClient
from src.llm.pricing import estimate_total_cost_cny
from src.llm.router import ModelTier, resolve_model
from src.reports.writer import write_final_summary

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"

REPORTS_DIR = Path("reports") / PROJECT_ID
BRIEF_PATTERN = re.compile(r"^(\d{2})_(.+)\.brief\.md$")


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1

    ctx = AgentContext(
        project_id=PROJECT_ID,
        ticker=TICKER,
        company_name=COMPANY,
        industry=INDUSTRY,
        reports_dir=REPORTS_DIR,
        rag=None,
    )

    # 按文件名排序加载 brief, 跳过 09_*.brief.md (decision 自身)
    loaded = []
    for f in sorted(REPORTS_DIR.iterdir()):
        m = BRIEF_PATTERN.match(f.name)
        if not m:
            continue
        step_no = int(m.group(1))
        agent_name = m.group(2)
        if step_no >= 9:
            continue
        ctx.briefs[agent_name] = f.read_text(encoding="utf-8")
        loaded.append((step_no, agent_name))

    logger.info(f"loaded {len(ctx.briefs)} briefs: {loaded}")

    llm = LLMClient()
    logger.info(f"DECIDE tier resolved -> {resolve_model(ModelTier.DECIDE)}")

    agent = DecisionAgent(llm)
    report = agent.run(ctx)
    logger.info(f"decision OK, brief: {report.brief[:200]}")

    agent._save_full_report(ctx, 9, report.full_report)
    (REPORTS_DIR / "09_decision.brief.md").write_text(
        report.brief, encoding="utf-8"
    )

    final_path = write_final_summary(ctx)
    logger.info(f"FINAL_MEMO written: {final_path}")

    # 落库到 feedback DB（Phase A 闭环学习需要）
    # 注意: rerun 时 ctx.extras.misc.score_cards 为空, 因前 8 步的原始 raw 已被 strip,
    # 无法从 .md 重建。下次走完整 cli analyze 流程才会有完整 score_cards。
    try:
        pid = persist_prediction(ctx, llm)
        if pid is not None:
            logger.info(f"prediction 已落库 id={pid}")
    except Exception as e:
        logger.warning(f"prediction 落库失败（不影响报告产出）: {e}")

    tier_to_model = {t.value: resolve_model(t) for t in ModelTier}
    cost = estimate_total_cost_cny(llm.ledger.by_tier, tier_to_model)
    (REPORTS_DIR / "_token_usage_decision_rerun.md").write_text(
        "# Token 使用账本（仅 decision 重跑）\n\n"
        + llm.ledger.summary()
        + f"\n\n**预估成本: ¥{cost}**（kimi-k2.x 不在 pricing 表里，会显示 0）\n\n"
        + "Tier → Model:\n"
        + "\n".join(f"- {t}: `{m}`" for t, m in tier_to_model.items()),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
