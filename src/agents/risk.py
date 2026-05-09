"""风控委员会 Agent。综合所有研究 + 辩论结论，从风险维度独立审视。"""
from __future__ import annotations

from src.agents._template import TemplateAgent, briefs_context
from src.agents.base import AgentContext
from src.feedback.models import RiskScoreCard
from src.llm import ModelTier

SYSTEM = """你是基石投资委员会的风控总监，独立于研究端。基于所有研究 Agent 简报和 Bull/Bear 辩论结论，
对以下风险维度进行独立评级（1=极高 / 5=极低）并给出风控决议：

## 一、信用与财务造假风险
## 二、行业逆风与赛道风险
## 三、估值高估与破发风险（重点关注 6 个月禁售期内）
## 四、流动性风险（上市后日均成交、自由流通股）
## 五、股东减持/解禁压力
## 六、监管与合规风险（中国/香港/海外）
## 七、关连交易与公司治理风险
## 八、ESG 实质性风险

最后给出：
## 综合风控评级（1-5）
## 否决条件（满足任一即否决基石认购）
## 限制性条件（认购金额上限、估值上限、必须的尽调补充）

输出 1500-2000 字，态度严谨保守。"""


class RiskAgent(TemplateAgent):
    name = "risk"
    description = "风控委员会 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = RiskScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        upstream = [
            "prospectus_analyst", "industry", "macro", "comparable",
            "tech_trend", "sentiment", "debate_manager",
        ]
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 全部研究简报与辩论裁决\n{briefs_context(ctx, upstream)}\n\n"
            f"请输出风控独立评估。"
        )
