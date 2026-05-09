"""Postmortem Agent: 对比 prediction vs outcome，输出结构化复盘 + memo。

设计要点:
1. 客观对比为先（确定性计算）：估值误差、风险命中数等用代码算，不让 LLM 推
2. 主观判断让 LLM 做：根因归类、Per-Agent 质量评分、教训提炼
3. 一次 LLM 调用同时产出 markdown memo + Score JSON
4. 自动把案例 upsert 到 CaseRAG（让下个项目能用上这次教训）

不能跑 Postmortem 的前置条件：必须有 outcome（否则没"实际结果"对比）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.feedback.models import Outcome, Prediction, Score
from src.feedback.store import FeedbackStore
from src.llm import LLMClient, ModelTier


POSTMORTEM_SYSTEM = """你是港股 IPO 基石投资委员会的复盘官。你的任务是客观、严谨地对比
"当时的决议"和"实际投后表现"，给出结构化复盘报告。

复盘原则:
1. 数据驱动，不情绪化；说"我们错了"必须给出具体证据
2. 区分"运气"和"判断"——一次决策结果好/坏不等于决策对/错
3. 根因归类必须严格（信息缺失/推理错误/估值方法不当/黑天鹅）
4. Per-Agent 质量打分要在评估到该 Agent 的输入信息可得性的基础上给

输入你将看到:
- 当时的决议（recommendation/估值区间/key_supports/key_risks/Per-Agent 评分卡）
- 实际投后表现（招股价/各 horizon 收益/是否破发/最大回撤等）
- 部分确定性指标已由代码预先算好（valuation_error_pct 等），你引用即可

输出严格按以下结构（JSON 在前，markdown 论述在后）:

```json
{
  "recommendation_score": <-1.0 ~ 1.0, 我们的建议事后看正确度>,
  "valuation_within_range": <true/false/null>,
  "valuation_error_pct": <数字 或 null, 已由代码预算>,
  "risks_total_count": <number>,
  "risks_realized_count": <number, 我们当时列的风险中实际兑现了几条>,
  "unforeseen_risks_count": <number, 实际发生但我们当时没识别的风险数>,
  "per_agent_quality": {
    "prospectus_analyst": <0-1>,
    "industry": <0-1>,
    "macro": <0-1>,
    "comparable": <0-1>,
    "tech_trend": <0-1>,
    "sentiment": <0-1>,
    "risk": <0-1>
  },
  "confidence_calibration_delta": <数字 或 null, 我们说"高/中/低"置信度时实际命中率差距>,
  "error_root_causes": [<"信息缺失"|"推理错误"|"估值方法不当"|"黑天鹅" 的子集>]
}
```

之后用以下 Markdown 章节:

## 一、客观结果对照
（罗列：建议 vs 结果、估值 vs 实际定价、风险 vs 兑现情况）

## 二、Per-Agent 质量评估
（每个 Agent 输出值不值，引用具体评分卡数据）

## 三、根因分析
（如果决议错了，错在哪？信息层 / 推理层 / 估值层 / 不可预测）

## 四、教训与可迁移经验
（写给"下一个类似项目分析"看：什么信号本次被忽视；什么估值方法本次失效；
未来类似行业/规模/估值倍数的项目，应额外关注什么）

第四节是这次复盘对系统的最大价值——它会被注入到未来类似项目的 prompt 里。"""


def _build_postmortem_input(p: Prediction, o: Outcome) -> tuple[str, dict]:
    """拼接给 LLM 的输入文本 + 预算好的确定性指标。"""
    # 估值误差（如果有 ipo_actual_marketcap）
    valuation_error_pct = None
    valuation_within_range = None
    if o.ipo_actual_marketcap_hkd_billion is not None and p.valuation_mid:
        valuation_error_pct = (
            (o.ipo_actual_marketcap_hkd_billion - p.valuation_mid) / p.valuation_mid
        )
        if p.valuation_low is not None and p.valuation_high is not None:
            valuation_within_range = (
                p.valuation_low <= o.ipo_actual_marketcap_hkd_billion <= p.valuation_high
            )

    pre_computed = {
        "valuation_error_pct": (
            round(valuation_error_pct, 4) if valuation_error_pct is not None else None
        ),
        "valuation_within_range": valuation_within_range,
        "risks_total_count": len(p.key_risks),
    }

    body = f"""# 当时的决议
- 项目: {p.ticker} {p.company_name}
- 行业: {p.industry}
- 决策日期: {p.decision_date:%Y-%m-%d}
- **建议**: {p.recommendation} (置信度: {p.confidence})
- **估值区间**: {p.valuation_low}-{p.valuation_high} 亿港元 (中枢 {p.valuation_mid}, 锚定 {p.anchor_method})
- 锚定逻辑: {p.anchor_logic}
- 定价观点: {p.ipo_pricing_view}
- 建议金额: {p.suggested_amount_low_usd_m}-{p.suggested_amount_high_usd_m} 万美元

## 当时的支持论据
{chr(10).join(f"- {s}" for s in p.key_supports)}

## 当时识别的风险
{chr(10).join(f"- {r}" for r in p.key_risks)}

## 当时设定的硬条件
{chr(10).join(f"- {c}" for c in p.deal_conditions)}

## 当时的 Agent 评分卡
{json.dumps(p.agent_score_cards, ensure_ascii=False, indent=2)}

# 实际投后表现
- 招股价: {o.ipo_actual_price_hkd} HKD
- 实际市值: {o.ipo_actual_marketcap_hkd_billion} 亿港元
- 上市日: {o.final_listing_date}
- D1 收益: {_pct(o.d1_return)}
- D30 收益: {_pct(o.d30_return)}
- D90 收益: {_pct(o.d90_return)}
- D180 收益（6 月禁售期满）: {_pct(o.d180_return)}
- D365 收益: {_pct(o.d365_return)}
- 是否首日破发: {o.was_broken_ipo_d1}
- 是否 6 月内破发: {o.was_broken_ipo_d180}
- 锁定期最大回撤: {_pct(o.max_drawdown_in_lockup_pct, 1)}
- 锁定期最低价: {o.min_price_in_lockup_hkd}
- D180 日均成交: {o.avg_daily_turnover_hkd_m_d180} 百万港元
- 基石实际认购: {o.cornerstone_actual_amount_usd_million} 万美元
- 解禁日实际收益: {_pct(o.cornerstone_realized_return_pct)}
- 用户备注: {o.user_notes}
- 重大事件: {o.notable_events}

# 已由代码预算的确定性指标
{json.dumps(pre_computed, ensure_ascii=False, indent=2)}
"""
    return body, pre_computed


def _pct(v: float | None, scale: float = 100.0) -> str:
    if v is None:
        return "—"
    return f"{v * scale:.2f}%"


def _parse_score_json(text: str) -> dict | None:
    matches = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    if not matches:
        return None
    try:
        return json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return None


class PostmortemAgent:
    """复盘 Agent。独立于 BaseAgent（不进 workflow，由 CLI 触发）。"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def run(self, prediction_id: int, store: FeedbackStore | None = None) -> Score | None:
        store = store or FeedbackStore()
        p = store.get_prediction(prediction_id)
        if p is None:
            logger.error(f"prediction id={prediction_id} not found")
            return None
        o = store.latest_outcome(prediction_id)
        if o is None:
            logger.error(f"prediction id={prediction_id} 还没有 outcome，无法复盘")
            return None

        body, pre_computed = _build_postmortem_input(p, o)
        resp = self.llm.complete(
            tier=ModelTier.DECIDE,  # 复盘也是核心判断，用 DECIDE tier
            system=POSTMORTEM_SYSTEM,
            messages=[{"role": "user", "content": body}],
            max_tokens=3500,
            temperature=0.2,
        )
        memo_text = resp.text
        parsed = _parse_score_json(memo_text)

        if parsed is None:
            logger.warning("Postmortem 输出未包含合规 JSON，仅记录 markdown")
            score = Score(
                prediction_id=prediction_id,
                score_date=datetime.now(),
                recommendation_score=0.0,
                postmortem_memo=memo_text,
            )
        else:
            # 把代码预算的字段覆盖（防止 LLM 重新算错）
            parsed["valuation_error_pct"] = pre_computed["valuation_error_pct"]
            parsed["valuation_within_range"] = pre_computed["valuation_within_range"]
            parsed["risks_total_count"] = pre_computed["risks_total_count"]
            parsed["postmortem_memo"] = memo_text
            parsed["prediction_id"] = prediction_id
            parsed["score_date"] = datetime.now()
            try:
                score = Score.model_validate(parsed)
            except ValidationError as e:
                logger.warning(f"Score schema 校验失败: {e}; fallback 用最小记录")
                score = Score(
                    prediction_id=prediction_id,
                    score_date=datetime.now(),
                    recommendation_score=float(parsed.get("recommendation_score", 0)),
                    postmortem_memo=memo_text,
                )

        store.save_score(score)

        # 复盘 memo 也落到对应项目的 reports 目录（如存在）
        try:
            from pathlib import Path
            if p.reports_dir_path:
                memo_path = Path(p.reports_dir_path) / "POSTMORTEM.md"
                memo_path.write_text(memo_text, encoding="utf-8")
                logger.info(f"复盘 memo 写入 {memo_path}")
        except Exception as e:
            logger.warning(f"写复盘 memo 文件失败: {e}")

        # 案例索引到 RAG（让未来类似项目用上）
        try:
            from src.feedback.case_rag import CaseRAG
            rag = CaseRAG()
            rag.index_case(p, o, score)
            logger.info(f"案例已索引到 CaseRAG: {p.project_id}")
        except Exception as e:
            logger.warning(f"案例索引失败: {e}")

        return score
