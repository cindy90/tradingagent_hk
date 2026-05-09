"""最终决策 Agent：基石投资认购建议。

特点：
- 用 Opus（最贵的 tier），因为是核心决策点；
- 输入只读各 Agent 的 brief 摘要（不读全文，省 token）；
- 输出严格结构化：包含投决建议 / 估值区间 / 关键依据 / 决议条件；
- Pydantic schema 校验：缺字段或格式错时自动重试一次，把 validation error
  附给 LLM 让其修正，避免拿到看似完整、实则关键字段缺失的"决议"。
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.llm import ModelTier

DECISION_SYSTEM = """你是港股 IPO 基石投资委员会的首席投决官，向最终基金 GP 汇报。
工作风格对标顶级投行 MD: 数据驱动、可证伪、可执行。

你将收到来自多个研究 Agent 的简报（包括基本面、行业、宏观、可比估值、技术趋势、情绪、Bull/Bear 辩论结论、风控意见）。
你的任务是：综合所有简报，给出最终基石投资建议。

注意：
- 基石投资有 6 个月禁售期，所以特别关注上市后 6-12 个月业绩兑现能力和后市流动性；
- 港股新股破发率历史上较高，估值锚定要保守；
- 必须区分"基本面投资价值" vs "基石认购的策略性价值"；
- 若某个 Agent 简报中明确标记"[执行失败]"，你必须在论述中显式承认对应信息缺失，
  并相应**降低 confidence**或要求**补充尽调**作为 deal_condition。

【关键约束 — 严禁幻觉】
- 你的输入只有各 Agent 的简报。**所有定量数字必须来自简报里出现过的数字**, 不要凭训练记忆补充。
  例如不要写"港股 IPO 平均 PS 18-22x"除非 macro / comparable 简报里有这个数字。
- 估值区间 (valuation_range_hkd_billion) 必须**从 comparable Agent 简报的 PS/PE/PB 锚定推算出来**,
  不能写"凭经验给 PS 8-10x"。anchor_logic 必须明确引用上游 Agent 的具体倍数。
- key_supports / key_risks **每条都必须能追溯到某个 Agent 简报**（如"现金 1480 万: prospectus_analyst"）。
- deal_conditions 必须**可量化、可触发**（"PS ≤ 20x 自动不认购" 优于 "估值过高即拒绝"）。
- monitoring_kpis 必须**可观测**（"Q1 经营现金流 vs 阈值" 优于 "管理层执行力"）。
- 提到具体公司时, 只允许使用上游 Agent 简报已用过的公司名。

【专业级新增要求】
1. **三档情景敏感性分析（必填 sensitivity_table）**:
   - "悲观" / "基准" / "乐观" 三档, 每档明确给出:
     * triggers: 1-3 条触发该情景的条件（必须可量化, 例如"H1 毛利率 < 30%" 而非"业绩疲弱"）
     * valuation_hkd_b: 该情景下的合理估值
     * probability: 主观概率 0-1, 三档之和应 ≈ 1.0
     * expected_return_pct: 该情景下相对招股价的回报率（%）
   - 这三档同时也是 Bull/Bear 辩论结论的"事后可验证清单"

2. **对冲与退出策略（hedging_strategy + exit_plan）**:
   - hedging_strategy: 锁定期 6 个月内的对冲思路（恒生科技 ETF PUT / 行业 long-short / 不对冲）
     * instrument / target_coverage_pct (0-1, 名义本金占比) / rationale
   - exit_plan: 解禁日的减持节奏
     * horizon ("解禁日 D+0" / "T+5") / method ("VWAP" / "市价" / "限价") / pace ("一次性" / "分 5 日") /
       trigger_conditions（提前减持的条件）

3. **kill_switches（认购后退出触发）**:
   - 已认购后某些事件应触发立刻减持/对冲, 列出 3-5 条:
     * trigger: 触发事件描述（必须可观测, 如"创始人或 CTO 任一离职 30 天内"）
     * action: 对应动作（"立刻减持 100%" / "对冲 30% 名义本金" / "暂停认购"）
     * severity: "高" / "中" / "低"

4. **monitoring_kpis 升级**:
   - 不再是文本列表, 而是结构化:
     * name (KPI 名) / threshold (阈值, 如"< 30%"或"> 200 天") /
       frequency ("季报" / "月报" / "事件触发") / action_if_breach（违阈后的动作）

输出严格按下述 JSON 结构（用 ```json``` 代码块包裹），之后再补一段中文论述（不超过 1200 字）：

```json
{
  "recommendation": "认购|审慎参与|观望|不认购",
  "confidence": "高|中|低",
  "suggested_amount_usd_million": [下限, 上限],
  "valuation_range_hkd_billion": {
    "low": <数字>, "mid": <数字>, "high": <数字>,
    "anchor_method": "PE|PS|EV/EBITDA|DCF|多方法加权|PEG",
    "anchor_logic": "<一句话锚定逻辑, 必须引用 comparable agent 的具体倍数>"
  },
  "ipo_pricing_view": "估值偏低|合理|偏高|严重高估",
  "sensitivity_table": [
    {
      "name": "悲观",
      "triggers": ["H1 毛利率 < 30%", "Q1 经营现金流转负"],
      "valuation_hkd_b": 45.0,
      "probability": 0.30,
      "expected_return_pct": -44.0
    },
    {"name": "基准", "triggers": [...], "valuation_hkd_b": 80.0, "probability": 0.50, "expected_return_pct": 0.0},
    {"name": "乐观", "triggers": [...], "valuation_hkd_b": 130.0, "probability": 0.20, "expected_return_pct": 62.5}
  ],
  "hedging_strategy": {
    "instrument": "恒生科技 ETF (3033.HK) PUT / 不对冲",
    "target_coverage_pct": 0.30,
    "rationale": "<为什么这样对冲, 1 句话>"
  },
  "exit_plan": {
    "horizon": "解禁日 D+0 / T+5",
    "method": "VWAP / 限价 / 市价",
    "pace": "一次性 / 分 5 日",
    "trigger_conditions": ["<提前减持的硬性条件, 如:解禁前股价跌破招股价 30%>"]
  },
  "kill_switches": [
    {"trigger": "创始人或 CTO 任一在 30 天内离职", "action": "立刻减持 100%", "severity": "高"},
    {"trigger": "监管问询函连续 2 次同类", "action": "减持 50% + 对冲剩余", "severity": "高"},
    {"trigger": "Q1 经营现金流连续两季为负", "action": "启动 ETF PUT 对冲", "severity": "中"}
  ],
  "key_supports": ["<3-5 条支持理由，每条不超过 30 字>"],
  "key_risks": ["<3-5 条核心风险，每条不超过 30 字>"],
  "deal_conditions": ["<对基石条款/估值的硬性要求, 必须可量化>"],
  "monitoring_kpis": ["<向后兼容的简短文本列表, 由 monitoring_kpis_detailed 自动生成>"],
  "monitoring_kpis_detailed": [
    {
      "name": "毛利率",
      "threshold": "< 30%",
      "frequency": "季报",
      "action_if_breach": "触发风险评估 + 考虑减持 30%"
    }
  ]
}
```

之后用以下 Markdown 章节展开论述（新增 4 节是专业级 IC memo 必备）：

## 一、投决论述
（综合论证，引用各 Agent 简报中的证据）

## 二、与 Bull/Bear 辩论的关系
（说明你采纳了哪一方哪些观点，为什么）

## 三、敏感性分析详解 ⭐
（针对悲观/基准/乐观三档, 解释 triggers 为何可量化, 估值如何反推, 概率如何主观赋值）

## 四、对冲与退出策略 ⭐
（解释 hedging_strategy 和 exit_plan 的逻辑, 为什么这种节奏 / 工具）

## 五、Kill Switches 触发条件 ⭐
（解释每条退出触发的合理性, 指出对应的 monitoring_kpis）

## 六、风险敞口与不确定性
（列出主要不确定性和应对方式, 引用风控简报）"""


_REC_VALUES = {"认购", "审慎参与", "观望", "不认购"}
_CONF_VALUES = {"高", "中", "低"}
_PRICING_VALUES = {"估值偏低", "合理", "偏高", "严重高估"}


class ValuationRange(BaseModel):
    low: float | None = None
    mid: float
    high: float | None = None
    anchor_method: str
    anchor_logic: str


# --- 专业级新增：敏感性 / 对冲 / 退出 / kill switches / 详细 KPI ---

class ScenarioRow(BaseModel):
    """敏感性分析的一行情景。"""
    model_config = ConfigDict(extra="ignore")
    name: str  # 悲观 / 基准 / 乐观（或更细分）
    triggers: list[str] = Field(default_factory=list, description="触发该情景的 1-3 条可量化条件")
    valuation_hkd_b: float
    probability: float = Field(ge=0, le=1)
    expected_return_pct: float | None = Field(default=None, description="该情景下相对招股价回报率")


class HedgingStrategy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instrument: str = "不对冲"
    target_coverage_pct: float = Field(default=0.0, ge=0, le=1, description="对冲名义本金占比")
    rationale: str = ""


class ExitPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    horizon: str = "解禁日 D+0"
    method: str = "VWAP"
    pace: str = "一次性"
    trigger_conditions: list[str] = Field(default_factory=list)


class KillSwitch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trigger: str
    action: str
    severity: Literal["高", "中", "低"] = "中"


class MonitoringKPI(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    threshold: str = ""
    frequency: str = "季报"
    action_if_breach: str = ""


class DecisionResult(BaseModel):
    """投决结果 schema。LLM 输出必须能转成本模型，否则触发重试。

    新增字段（专业级 IC memo 标志）：
    - sensitivity_table: 三档情景敏感性
    - hedging_strategy / exit_plan: 锁定期对冲 + 解禁退出
    - kill_switches: 认购后退出触发
    - monitoring_kpis_detailed: 阈值化 KPI（旧 monitoring_kpis 文本列表向后兼容）
    """
    model_config = ConfigDict(extra="ignore")

    recommendation: str = Field(description="认购/审慎参与/观望/不认购")
    confidence: str = Field(description="高/中/低")
    suggested_amount_usd_million: list[float] = Field(min_length=2, max_length=2)
    valuation_range_hkd_billion: ValuationRange
    ipo_pricing_view: str
    key_supports: list[str] = Field(min_length=1)
    key_risks: list[str] = Field(min_length=1)
    deal_conditions: list[str] = Field(default_factory=list)
    monitoring_kpis: list[str] = Field(default_factory=list)

    # 专业级新增字段（向后兼容：默认 None / 空 list）
    sensitivity_table: list[ScenarioRow] = Field(
        default_factory=list,
        description="三档情景敏感性（悲观/基准/乐观），probability 之和应 ≈ 1",
    )
    hedging_strategy: HedgingStrategy | None = None
    exit_plan: ExitPlan | None = None
    kill_switches: list[KillSwitch] = Field(default_factory=list)
    monitoring_kpis_detailed: list[MonitoringKPI] = Field(default_factory=list)


def _briefs_block(briefs: dict[str, str]) -> str:
    if not briefs:
        return "（无可用简报）"
    parts = []
    for name, brief in briefs.items():
        parts.append(f"### Agent: `{name}`\n{brief}")
    return "\n\n".join(parts)


def parse_decision_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中抽取 ```json``` 代码块；解析失败返回 None。"""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def validate_decision(parsed: dict[str, Any] | None) -> tuple[DecisionResult | None, str]:
    """返回 (model, error_message)；error_message 非空表示校验失败。"""
    if parsed is None:
        return None, "无法从输出中提取 ```json``` 代码块。"
    try:
        result = DecisionResult.model_validate(parsed)
    except ValidationError as e:
        return None, str(e)
    # 软校验：枚举值不强制，但记录警告
    warnings: list[str] = []
    if result.recommendation not in _REC_VALUES:
        warnings.append(f"recommendation 应为 {_REC_VALUES}，得到 '{result.recommendation}'")
    if result.confidence not in _CONF_VALUES:
        warnings.append(f"confidence 应为 {_CONF_VALUES}，得到 '{result.confidence}'")
    if result.ipo_pricing_view not in _PRICING_VALUES:
        warnings.append(f"ipo_pricing_view 应为 {_PRICING_VALUES}，得到 '{result.ipo_pricing_view}'")
    return result, "; ".join(warnings)


class DecisionAgent(BaseAgent):
    name = "decision"
    tier = ModelTier.DECIDE
    description = "最终基石投资决策 Agent"
    fatal = True  # 决策失败 = 没有产出，整个流程白跑

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
        parsed = parse_decision_json(full)
        result, err = validate_decision(parsed)

        # 第一次校验失败：把 error message 塞回给 LLM 让它重试一次
        if result is None:
            logger.warning(f"决议 JSON 校验失败，触发一次重试。错误: {err[:300]}")
            retry_msg = (
                f"上一轮输出的 JSON 不符合 schema，校验报错如下：\n\n```\n{err[:1000]}\n```\n\n"
                f"请重新生成完整的 ```json``` 代码块，**确保所有必填字段齐全且类型正确**：\n"
                f"- recommendation / confidence / ipo_pricing_view 必须是字符串\n"
                f"- suggested_amount_usd_million 必须是恰好 2 个数字的数组 [下限, 上限]\n"
                f"- valuation_range_hkd_billion 必须包含 mid / anchor_method / anchor_logic\n"
                f"- key_supports / key_risks 至少 1 项\n"
                f"重新输出完整结果（含 JSON + 论述章节）。"
            )
            resp2 = self.llm.complete(
                tier=self.tier,
                system=DECISION_SYSTEM,
                messages=[
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": full},
                    {"role": "user", "content": retry_msg},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            full = resp2.text
            parsed = parse_decision_json(full)
            result, err = validate_decision(parsed)
            if result is None:
                logger.error(f"决议 JSON 重试仍失败: {err[:300]}")

        # 构造给人看的 brief（也是 ledger 落盘的前 N 字）
        if result is not None:
            valuation = result.valuation_range_hkd_billion
            brief = (
                f"**最终建议**: {result.recommendation} (置信度: {result.confidence})\n"
                f"**估值区间**: {valuation.low}-{valuation.high} (中枢 {valuation.mid}) "
                f"亿港元 / {valuation.anchor_method}\n"
                f"**定价观点**: {result.ipo_pricing_view}"
            )
            if err:
                brief += f"\n*Schema 警告*: {err[:200]}"
        else:
            brief = (
                "**[决议 JSON 校验失败]** 经过一次重试仍未拿到合规决议。"
                "请人工审阅完整报告。"
            )

        ctx.full_reports[self.name] = full
        ctx.briefs[self.name] = brief
        ctx.extras.decision_json = (
            result.model_dump() if result is not None else parsed
        )

        return AgentReport(
            agent=self.name,
            full_report=full,
            brief=brief,
            metadata={"decision_result": result.model_dump() if result else None,
                      "validated": result is not None,
                      "schema_warnings": err if result else None},
        )
