from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.data.akshare_client import get_hsi_index
from src.feedback.models import MacroScoreCard
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
    score_card_class = MacroScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        hsi = get_hsi_index()
        macro = ctx.extras.macro_indicators

        # 把 EDB 时间序列压缩为"最新值 / 6 个月前 / 12 个月前 / 趋势"，省 token
        macro_lines = []
        for name, series in macro.items():
            if not series:
                macro_lines.append(f"- {name}: (无数据)")
                continue
            latest = series[-1]
            prev6 = series[-min(6, len(series))] if len(series) > 1 else None
            prev12 = series[-min(12, len(series))] if len(series) > 1 else None
            parts = [f"最新({latest.get('date')}): {latest.get('value')}"]
            if prev6:
                parts.append(f"6 期前({prev6.get('date')}): {prev6.get('value')}")
            if prev12 and prev12 is not prev6:
                parts.append(f"12 期前({prev12.get('date')}): {prev12.get('value')}")
            macro_lines.append(f"- {name}: " + " | ".join(parts))
        macro_block = "\n".join(macro_lines) if macro_lines else "(同花顺 EDB 未返回数据)"

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 当前恒生指数（akshare）\n{hsi}\n\n"
            f"# 宏观指标（同花顺 EDB）\n{macro_block}\n\n"
            f"请输出宏观策略分析。"
        )
