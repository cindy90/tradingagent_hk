from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import SentimentScoreCard
from src.llm import ModelTier

SYSTEM = """你是港股二级市场情绪分析师。基于近期市场表现、新股暗盘情况、同行业公司股价走势，输出：

## 一、所在板块近期情绪（涨跌、成交、北水流向）
## 二、可比已上市公司股价表现（近 30/90 天）
## 三、近期同行业 IPO 暗盘/首日表现
## 四、市场关注度迹象（媒体覆盖、分析师跟踪）
## 五、对本次基石认购的情绪面判断（顺风/逆风）

输出 1000-1500 字。如缺乏数据，明确指出数据缺口。"""


class SentimentAgent(TemplateAgent):
    name = "sentiment"
    description = "二级市场情绪 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = SentimentScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        peers_quotes = ctx.extras.peer_recent_quotes
        recent_ipos = ctx.extras.recent_hk_ipos
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 可比公司近期行情\n{peers_quotes}\n\n"
            f"# 近期同行业 IPO 表现\n{recent_ipos}\n\n"
            f"请输出情绪分析。"
        )
