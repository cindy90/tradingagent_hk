from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.data.akshare_client import get_hsi_index
from src.llm import ModelTier

SYSTEM = """你是港股市场宏观策略分析师。基于当前宏观环境，分析对本 IPO 基石认购的影响：

## 一、全球与中国宏观（美联储路径、人民币汇率、中国经济周期）
## 二、香港市场流动性（HIBOR、IPO 集资额、北水南下情况、新股认购热度）
## 3、恒生指数估值与情绪状态（PE 分位、波动率）
## 四、港股 IPO 市场近期表现（破发率、首日涨跌、暗盘表现）
## 五、对本项目所属板块的偏好倾向
## 六、宏观环境对基石策略的总体判断（窗口好/坏，1-5 分）

要求：使用提供的市场数据，给出有数字支撑的判断。输出 1200-1800 字。"""


class MacroAgent(TemplateAgent):
    name = "macro"
    description = "宏观策略 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        hsi = get_hsi_index()
        macro = ctx.extras.get("macro_indicators", {})
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 当前恒生指数\n{hsi}\n\n"
            f"# 宏观指标（同花顺/外部）\n{macro}\n\n"
            f"请输出宏观策略分析。"
        )
