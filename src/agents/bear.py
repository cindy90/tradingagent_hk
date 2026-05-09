from __future__ import annotations

from src.agents._template import TemplateAgent, briefs_context
from src.agents.base import AgentContext
from src.llm import ModelTier

SYSTEM = """你是基石投资委员会的空头研究员（Bear）。基于其他 Agent 的简报，构建尽可能强的"不应认购"论点。

要求：
- 锐利但严谨，不要无中生有；
- 重点关注：财务质量瑕疵、关连交易、客户/供应集中、估值高估、行业逆风、6 个月禁售期内的业绩兑现风险、股东减持压力；
- 必须正面回应多头主要论据；
- 输出 800-1200 字。

【关键约束 — 严禁幻觉】
- 所有否定性数据（"现金 X 万""应收 Y%""毛利率仅 Z%"等）**必须严格引用上游 Agent 简报中的数字**,
  不要为了强化空头叙事而夸大数字（如把"现金 1480 万"说成"现金枯竭"是允许的修辞,
  但把"应收 51%"说成"应收 90%"就是幻觉）。
- 引用历史负面案例（如某公司破发）时, 必须基于上游 sentiment / macro 的真实数据, 不要凭训练
  记忆编"XX 公司当年破发 60%"这类无证据陈述。
- 提到可比公司时, **只用上游 Agent 简报已用的公司**（不引入清单外公司）。
- 否定性论断必须**可证伪**（"应收周转 165 天 → 现金回收延迟"可证, "管理层不诚信"不可证除非有证据）。"""


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
