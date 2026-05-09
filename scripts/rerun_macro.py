"""一次性脚本: 用修复后的 EDB 解析 + 真实 HIBOR 编码 + 防幻觉 prompt 重跑 macro agent.

验证目的:
1. ths.edb() 真返回数据（之前因 _parse_edb_payload bug 一直返回空）
2. macro 报告引用真实 HIBOR 数字（不再写"2024 年 HIBOR 3.5-4.0%"）
3. prompt 防幻觉约束生效（缺数据时不编年份）

不动其他 agent，只覆盖 03_macro.md / 03_macro.brief.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from dotenv import load_dotenv
load_dotenv()

from src.agents.base import AgentContext
from src.agents.extras import WorkflowExtras
from src.agents.macro import MacroAgent
from src.agents.summarizer import Summarizer
from src.data.ths_client import THSClient
from src.graph.workflow import _prefetch_ifind_sdk
from src.llm import LLMClient

# 近期同行业 IPO 队列, 用于计算破发率（与 sentiment 共用）
DEFAULT_RECENT_IPOS = ["02432", "01021", "09880", "09660"]

PROJECT_ID = "02670_20260509_135040"
TICKER = "02670"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"

REPORTS_DIR = Path("reports") / PROJECT_ID


def main() -> int:
    if not REPORTS_DIR.exists():
        logger.error(f"找不到 {REPORTS_DIR}")
        return 1

    # 拿真实宏观指标
    ths = THSClient()
    if not ths.configured:
        logger.error("THS 未配置（IFIND_REFRESH_TOKEN 缺失）")
        return 1
    macro_data = ths.edb()
    non_empty = {k: v for k, v in macro_data.items() if v}
    logger.info(f"EDB 返回 {len(non_empty)} 个非空指标: {list(non_empty.keys())}")
    for name, series in non_empty.items():
        logger.info(
            f"  {name}: {len(series)} 期 / 最新 {series[-1]['date']} -> {series[-1]['value']}"
        )

    extras = WorkflowExtras()
    extras.macro_indicators = macro_data
    # 拉近期 IPO 队列, 给 macro 算破发率用
    sdk_data = _prefetch_ifind_sdk(DEFAULT_RECENT_IPOS, target_ticker="H2254", expected_target_name=COMPANY)
    extras.recent_hk_ipos = sdk_data["recent_hk_ipos"]
    logger.info(f"recent_hk_ipos 注入 {len(extras.recent_hk_ipos)} 条")

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
    agent = MacroAgent(llm, Summarizer(llm))
    report = agent.run(ctx)
    logger.info(f"macro brief (前 200 字): {report.brief[:200]}")

    agent._save_full_report(ctx, 3, report.full_report)
    (REPORTS_DIR / "03_macro.brief.md").write_text(
        report.brief, encoding="utf-8"
    )
    logger.info(f"已重写 {REPORTS_DIR}/03_macro.md 和 03_macro.brief.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
