from __future__ import annotations

from src.agents._template import TemplateAgent, briefs_context
from src.agents.base import AgentContext
from src.llm import ModelTier

SYSTEM = """你是基石投资委员会的空头研究员（Bear）。基于其他 Agent 的简报，构建尽可能强的"不应认购"论点。

要求：
- 锐利但严谨，不要无中生有；
- 重点关注：财务质量瑕疵、关连交易、客户/供应集中、估值高估、行业逆风、6 个月禁售期内的业绩兑现风险、股东减持压力；
- 必须正面回应多头主要论据；
- 输出 800-1200 字。"""


class BearResearcher(TemplateAgent):
    name = "bear"
    description = "Bear 研究员（空头观点）"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        upstream = ["prospectus_analyst", "industry", "macro", "comparable", "tech_trend", "sentiment"]
        bull_view = ctx.briefs.get("bull", "")
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 上游研究简报\n{briefs_context(ctx, upstream)}\n\n"
            f"# 对手方（Bull）当前观点\n{bull_view or '（首轮，无）'}\n\n"
            f"请构建空头观点。"
        )
