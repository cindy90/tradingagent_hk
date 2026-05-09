from __future__ import annotations

from src.agents._template import TemplateAgent, briefs_context
from src.agents.base import AgentContext
from src.llm import ModelTier

SYSTEM = """你是基石投资委员会的多头研究员（Bull）。基于其他 Agent 的简报，构建尽可能强的"应该认购"论点。

要求：
- 引用其他 Agent 简报里的具体数据/事实作为论据；
- 必须正面回应明显存在的反对观点（不是回避）；
- 输出 800-1200 字，分论点 + 证据；
- 最后给出"在何种条件下变得不应认购"的反向触发条件，体现思辨。"""


class BullResearcher(TemplateAgent):
    name = "bull"
    description = "Bull 研究员（多头观点）"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        upstream = ["prospectus_analyst", "industry", "macro", "comparable", "tech_trend", "sentiment"]
        bear_view = ctx.briefs.get("bear", "")
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 上游研究简报\n{briefs_context(ctx, upstream)}\n\n"
            f"# 对手方（Bear）当前观点\n{bear_view or '（首轮，无）'}\n\n"
            f"请构建多头观点。"
        )
