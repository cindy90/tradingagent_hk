from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.llm import ModelTier
from src.tools.valuation import comparable_valuation

SYSTEM = """你是港股 IPO 估值分析师，专门做可比公司估值。基于：
- 目标公司财务数据（净利润、营收、EBITDA）
- 已计算好的可比公司估值倍数与目标估值区间

输出：

## 一、可比公司选择依据（行业、规模、商业模式契合度）
## 二、估值倍数对比表（PE / PS / EV-EBITDA）
## 三、目标公司估值区间（低 / 中 / 高三档，分别对应何种倍数水平）
## 四、与招股书招股价区间对比（如有）
## 五、估值锚定结论（建议基石可接受估值上限）

注意：
- 不要自己重新算数，使用工具传来的计算结果；
- 港股 IPO 通常会比可比公司有 10-25% 的 IPO 折扣。

输出 1200-1500 字。"""


class ComparableAgent(TemplateAgent):
    name = "comparable"
    description = "可比公司估值 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        peers = ctx.extras.get("peers", [])
        target_metric = ctx.extras.get("target_net_profit") or ctx.extras.get("target_revenue") or 0
        peer_pe = ctx.extras.get("peer_pe_multiples", [])
        peer_ps = ctx.extras.get("peer_ps_multiples", [])

        pe_val = comparable_valuation(target_metric, peer_pe, "PE") if peer_pe else {}
        ps_val = comparable_valuation(target_metric, peer_ps, "PS") if peer_ps else {}

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 可比公司列表\n{peers}\n\n"
            f"# PE 估值结果\n{pe_val}\n\n"
            f"# PS 估值结果\n{ps_val}\n\n"
            f"请输出可比公司估值分析。"
        )
