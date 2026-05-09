"""最终决策 Agent：基石投资认购建议。

特点：
- 用 Opus（最贵的 tier），因为是核心决策点；
- 输入只读各 Agent 的 brief 摘要（不读全文，省 token）；
- 输出严格结构化：包含投决建议 / 估值区间 / 关键依据 / 决议条件。
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.llm import ModelTier

DECISION_SYSTEM = """你是港股 IPO 基石投资委员会的首席投决官，向最终基金 GP 汇报。

你将收到来自多个研究 Agent 的简报（包括基本面、行业、宏观、可比估值、技术趋势、情绪、Bull/Bear 辩论结论、风控意见）。
你的任务是：综合所有简报，给出最终基石投资建议。

注意：
- 基石投资有 6 个月禁售期，所以特别关注上市后 6-12 个月业绩兑现能力和后市流动性；
- 港股新股破发率历史上较高，估值锚定要保守；
- 必须区分"基本面投资价值" vs "基石认购的策略性价值"。

输出严格按下述 JSON 结构（用 ```json``` 代码块包裹），之后再补一段中文论述（不超过 800 字）：

```json
{
  "recommendation": "认购|审慎参与|观望|不认购",
  "confidence": "高|中|低",
  "suggested_amount_usd_million": [下限, 上限],
  "valuation_range_hkd_billion": {
    "low": <数字>,
    "mid": <数字>,
    "high": <数字>,
    "anchor_method": "PE|PS|EV/EBITDA|DCF|多方法加权",
    "anchor_logic": "<一句话锚定逻辑>"
  },
  "ipo_pricing_view": "估值偏低|合理|偏高|严重高估",
  "key_supports": ["<3-5 条支持理由，每条不超过 30 字>"],
  "key_risks": ["<3-5 条核心风险，每条不超过 30 字>"],
  "deal_conditions": ["<对基石条款/估值的硬性要求，例如最高可接受估值上限>"],
  "monitoring_kpis": ["<上市后需重点跟踪的 3-5 个 KPI>"]
}
```

之后用以下 Markdown 章节展开论述：

## 投决论述
（综合论证，引用各 Agent 简报中的证据）

## 与 Bull/Bear 辩论的关系
（说明你采纳了哪一方哪些观点，为什么）

## 风险敞口与对冲思路
（列出主要不确定性和应对方式）"""


def _briefs_block(briefs: dict[str, str]) -> str:
    if not briefs:
        return "（无可用简报）"
    parts = []
    for name, brief in briefs.items():
        parts.append(f"### Agent: `{name}`\n{brief}")
    return "\n\n".join(parts)


def parse_decision_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


class DecisionAgent(BaseAgent):
    name = "decision"
    tier = ModelTier.DECIDE
    description = "最终基石投资决策 Agent"

    def run(self, ctx: AgentContext) -> AgentReport:
        briefs_text = _briefs_block(ctx.briefs)

        user_msg = (
            f"# 待决策项目\n"
            f"- 公司：{ctx.company_name} ({ctx.ticker})\n"
            f"- 行业：{ctx.industry}\n"
            f"- 项目 ID：{ctx.project_id}\n\n"
            f"# 各 Agent 简报\n\n{briefs_text}\n\n"
            f"请按系统指令的格式输出最终基石投资决策。"
        )

        resp = self.llm.complete(
            tier=self.tier,
            system=DECISION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=4000,
            temperature=0.1,
        )
        full = resp.text
        decision_json = parse_decision_json(full)

        # 决策报告本身就是给人看的最终产物，brief 直接用 JSON 抽取
        if decision_json:
            brief = (
                f"**最终建议**: {decision_json.get('recommendation', 'N/A')} "
                f"(置信度: {decision_json.get('confidence', 'N/A')})\n"
                f"**估值区间**: {decision_json.get('valuation_range_hkd_billion', {})}\n"
                f"**定价观点**: {decision_json.get('ipo_pricing_view', 'N/A')}"
            )
        else:
            brief = full[:500]

        ctx.full_reports[self.name] = full
        ctx.briefs[self.name] = brief
        ctx.extras.decision_json = decision_json

        return AgentReport(
            agent=self.name,
            full_report=full,
            brief=brief,
            metadata={"decision_json": decision_json},
        )
