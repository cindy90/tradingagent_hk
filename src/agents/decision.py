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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

【专业级新增要求 — 必须输出"推理链"让用户能 trace 每个量化结论】

1. **三档情景敏感性分析（必填 sensitivity_table）**:
   - "悲观" / "基准" / "乐观" 三档, 每档明确给出:
     * triggers: 1-3 条触发该情景的条件（必须可量化, 例如"H1 毛利率 < 30%" 而非"业绩疲弱"）
     * valuation_hkd_b: 该情景下的合理估值
     * probability: 主观概率 0-1, 三档之和应 ≈ 1.0
     * expected_return_pct: 该情景下相对招股价的回报率（%）
     * **valuation_derivation**: 估值如何从 triggers 推算出的（必填, 一句话公式）
       例: "毛利率从 38% 降至 30% → PS 压缩 25% → 80×0.75≈60, 再扣行业逆风折让 25% = 45"
     * **probability_rationale**: 概率赋值依据（必填, 必须有数据基础）
       例: "近 3 月港股机器人 IPO 6 个月内破发率 30%; 加上本项目客户集中度高调高 5pp 至 35%"
   - 这三档同时也是 Bull/Bear 辩论结论的"事后可验证清单"

2. **估值方法分拆（valuation_range_hkd_billion.methodology_breakdown）⭐**:
   - 至少给 2-3 种估值方法的具体推算（PE / PS / PEG / EV-EBITDA / SOTP / DCF）
   - 每种方法填:
     * peer_basis: "可比 X PS 中位数 22x (越疆 27x / 华沿 18x)"  (必须引用 comparable 简报具体数字)
     * target_metric: "2025E 营收 3.6 亿 RMB" (必须明确口径 + 来源)
     * formula: "22 × 3.6 = 79.2 亿 RMB ≈ 88 亿 HKD (按 1.1 汇率)"
     * result_hkd_b: 88
     * weight: 该方法在加权综合估值中的权重 (各方法权重之和应 ≈ 1)
   - 这让审计能直接看到"22x 哪来的""3.6 亿哪来的""为什么不用 PE"
   - **valuation_range_hkd_billion.key_assumptions**: 估值依赖的 3-5 条关键假设
     例: ["2025E 营收 3.6 亿（+38% YoY）", "毛利率维持 38%", "可比 PS 中位 22x 不发生 25%+ 压缩"]

3. **对冲与退出策略（hedging_strategy + exit_plan）**:
   - hedging_strategy: 锁定期 6 个月内的对冲思路（恒生科技 ETF PUT / 行业 long-short / 不对冲）
   - exit_plan: 解禁日的减持节奏

4. **kill_switches（认购后退出触发）⭐ 每条必须有 rationale**:
   - 列出 3-5 条:
     * trigger: 触发事件描述（必须可观测, 如"创始人或 CTO 任一离职 30 天内"）
     * action: 对应动作（"立刻减持 100%" / "对冲 30% 名义本金" / "暂停认购"）
     * severity: "高" / "中" / "低"
     * **rationale: 为什么这个阈值**（必填, 不能空）
       例: "创始人股权占比 35%, 历史 IPO 案例显示创始人 30 天内离职后股价中位回撤 40%"
     * historical_precedent: 可选 — 相似案例

5. **monitoring_kpis_detailed（每条加 threshold_rationale + industry_benchmark）⭐**:
   - name (KPI 名) / threshold (如 "< 30%") / frequency / action_if_breach
   - **threshold_rationale: 为什么阈值是这个数**
     例: "毛利率行业中位 35%, 招股书披露 38%, 留 8pp 缓冲设阈值 30%"
   - **industry_benchmark**: 可选 — 行业对比基准

6. **key_assumptions（投决依赖的核心假设, 5-10 条）⭐**:
   - 把整个推理依赖的"如果不对就要重新评估"的假设列出来
   - 每条须可量化、可证伪
   - 例:
     ["2025E 营收 3.6 亿 RMB (+38% YoY) — 来自 prospectus_analyst 财务预测",
      "毛利率维持 38% ± 3pp (vs 行业中位 35%)",
      "可比公司 PS 中位 22x 在锁定期内不发生 > 25% 压缩",
      "无重大监管变动 (中美博弈不升级 / 香港数据出境政策稳定)",
      "创始人 / 核心管理层在锁定期内不离职"]

7. **reasoning_chain（5-10 步关键推理链）⭐⭐**:
   - 这是整份 IC memo 的灵魂——把"如何从一堆 Agent 简报推到最终建议"的链条显式写出
   - 每步含: step_no / title / premise / data_source / calculation / conclusion / confidence / caveats
   - 推理顺序通常是:
     步 1: 业务质量评估 (来自 prospectus_analyst)
     步 2: 行业空间与公司定位 (来自 industry)
     步 3: 估值起点 — 可比公司 PS 锚定 (来自 comparable)
     步 4: 估值修正 — IPO 折扣 / 增长溢价 (PEG)
     步 5: 风险约束 (来自 risk Agent)
     步 6: 宏观窗口判断 (来自 macro)
     步 7: 综合三档情景 → 期望回报
     步 8: 与 Bull/Bear 辩论结论核对 (来自 debate_manager)
     步 9: 数字交叉核对 (来自 fact_check)
     步 10: 最终建议
   - 每步的 calculation 必须给具体公式或文字描述, 不能是"综合考虑"
   - confidence 反映该步骤的把握度 (高/中/低)
   - caveats: 该步骤潜在异议, 例:"假设 PS 22x 不变, 但若行业受冲击可能压缩到 15x"

8. **decision_weights（决策因子加权打分卡, 5-7 个因子）⭐⭐⭐**:
   - 这是把"多 Agent 简报"综合到"最终建议"的**显式权重机制**
   - 不是 reasoning_chain 的替代, 是补充: 推理链解释"逻辑推导", 打分卡解释"加权综合"
   - **必须输出 5-7 个因子**, 推荐使用以下 6 个标准因子集合 (可根据项目特征微调权重):
     | 因子 | 建议权重区间 | 主要数据源 Agent |
     |---|---|---|
     | 业务质量 | 0.15-0.30 | prospectus_analyst, tech_trend |
     | 估值合理性 | 0.20-0.35 | comparable |
     | 风控等级 | 0.20-0.30 | risk |
     | 宏观与行业窗口 | 0.05-0.20 | macro, industry |
     | 辩论倾向 | 0.10-0.20 | debate_manager |
     | 情绪与流动性 | 0.05-0.15 | sentiment |
   - **每个因子必须填**:
     * factor: 因子名 (使用上表中的标准名, 或更细分但要明确)
     * weight: 0-1, 各因子之和应 ≈ 1.0 (允许 ±5% 偏差)
     * score: 0-5, 该因子当前得分 (5=最优, 0=最差)
     * source_agents: 数据源 Agent 名列表
     * rationale: **必填**, 必须同时说明
       (a) 为什么权重是这个数 (vs 标准区间的 lower/upper bound)
       (b) 为什么得分是这个数 (引用具体 Agent 简报的关键事实)
   - **加权总分 = Σ(weight × score)**, 系统会自动算
   - **映射到推荐档**: ≥4.0 认购 / 3.0-4.0 审慎参与 / 2.0-3.0 观望 / <2.0 不认购
   - **关键: 加权总分对应的推荐必须与你最终的 recommendation 字段一致**.
     如果加权总分 = 3.5 但你的 recommendation 是"认购", 必须在论述中说明
     "尽管加权 3.5 落在审慎参与区间, 但因 X 因素提升至认购"——否则视为前后矛盾.
   - **可调整权重的情形**: 例如对早期成长股可上调"业务质量"权重至 0.30; 对周期股
     上调"宏观窗口"权重至 0.20. 但每次调整都必须在 rationale 里明确说明.

输出严格按下述 JSON 结构（用 ```json``` 代码块包裹），之后再补一段中文论述（不超过 1200 字）：

```json
{
  "recommendation": "认购|审慎参与|观望|不认购",
  "confidence": "高|中|低",
  "suggested_amount_usd_million": [下限, 上限],
  "valuation_range_hkd_billion": {
    "low": <数字>, "mid": <数字>, "high": <数字>,
    "anchor_method": "PE|PS|EV/EBITDA|DCF|多方法加权|PEG",
    "anchor_logic": "<一句话锚定逻辑, 必须引用 comparable agent 的具体倍数>",
    "methodology_breakdown": [
      {"method": "PS", "peer_basis": "可比 PS 中位 22x (越疆 27/华沿 18)",
       "target_metric": "2025E 营收 3.6 亿 RMB", "formula": "22 × 3.6 ≈ 79 亿 RMB ≈ 88 亿 HKD",
       "result_hkd_b": 88.0, "weight": 0.5},
      {"method": "PEG", "peer_basis": "可比 PEG 中位 1.4x (按 35% CAGR)",
       "target_metric": "PE × CAGR = 70 × 0.35", "formula": "1.4 × 净利润 0.7 亿 RMB × 70 = ...",
       "result_hkd_b": 70.0, "weight": 0.3},
      {"method": "SOTP", "peer_basis": "硬件本体 PS 5x + 协作 PS 12x + 具身 PS 25x",
       "target_metric": "三业务收入 1.5/1.2/0.9 亿", "formula": "1.5×5+1.2×12+0.9×25=44.4 亿 RMB",
       "result_hkd_b": 49.0, "weight": 0.2}
    ],
    "key_assumptions": ["2025E 营收 3.6 亿 (+38% YoY)", "毛利率维持 38%",
                        "可比 PS 不发生 > 25% 压缩"]
  },
  "ipo_pricing_view": "估值偏低|合理|偏高|严重高估",
  "sensitivity_table": [
    {
      "name": "悲观", "triggers": ["H1 毛利率 < 30%", "Q1 经营现金流转负"],
      "valuation_hkd_b": 45.0, "probability": 0.30, "expected_return_pct": -44.0,
      "valuation_derivation": "毛利从 38% → 30%, PS 压缩 25% → 80×0.75=60; 再扣行业逆风折让 25% = 45",
      "probability_rationale": "近 3 月港股机器人 IPO 6 月内破发率 30%; 客户集中度高调高 5pp 至 35%, 取均值 30%"
    },
    {"name": "基准", "triggers": ["持平"], "valuation_hkd_b": 80.0, "probability": 0.50, "expected_return_pct": 0.0,
     "valuation_derivation": "可比 PS 中位 22x × 2025E 营收 3.6 亿 ≈ 79 亿 RMB ≈ 88 亿 HKD, 加权降至 80",
     "probability_rationale": "财务符合招股书披露 + 无极端事件; 概率赋值 50% 居中"},
    {"name": "乐观", "triggers": ["第二曲线兑现"], "valuation_hkd_b": 130.0, "probability": 0.20, "expected_return_pct": 62.5,
     "valuation_derivation": "具身智能业务收入占比从 9% 升至 18%, SOTP 估值贡献 +50% → 130 亿",
     "probability_rationale": "招股书披露 H2 大客户订单, 但兑现概率经验值 20-30%, 取下沿 20%"}
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
    {"trigger": "创始人或 CTO 任一在 30 天内离职", "action": "立刻减持 100%", "severity": "高",
     "rationale": "创始人股权 35%, 历史 IPO 案例创始人 30 天离职股价中位回撤 40%",
     "historical_precedent": "2024 年 X 公司 CTO 离职 D+15 股价跌 38%"},
    {"trigger": "监管问询函连续 2 次同类", "action": "减持 50% + 对冲剩余", "severity": "高",
     "rationale": "连续问询暗示实质合规问题, 2 次同类是显著信号"},
    {"trigger": "Q1 经营现金流连续两季为负", "action": "启动 ETF PUT 对冲", "severity": "中",
     "rationale": "本项目财务质量是基石认购核心假设, 现金流恶化触发风险敞口控制"}
  ],
  "key_supports": ["<3-5 条支持理由，每条不超过 30 字>"],
  "key_risks": ["<3-5 条核心风险，每条不超过 30 字>"],
  "deal_conditions": ["<对基石条款/估值的硬性要求, 必须可量化>"],
  "monitoring_kpis": ["<向后兼容的简短文本列表, 由 monitoring_kpis_detailed 自动生成>"],
  "monitoring_kpis_detailed": [
    {"name": "毛利率", "threshold": "< 30%", "frequency": "季报",
     "action_if_breach": "触发风险评估 + 考虑减持 30%",
     "threshold_rationale": "招股书披露 38% / 行业中位 35%, 30% 是行业 10 分位 = 显著恶化",
     "industry_benchmark": "行业中位 35%, 75 分位 42%, 25 分位 28%"}
  ],
  "key_assumptions": [
    "2025E 营收 3.6 亿 RMB (+38% YoY) — 招股书 P.108 财务预测",
    "毛利率维持 38% ± 3pp (vs 行业中位 35%)",
    "可比公司 PS 中位 22x 在锁定期内不发生 > 25% 压缩",
    "无重大监管变动 (中美博弈不升级 / 香港数据出境政策稳定)",
    "创始人 + CTO 在锁定期内不离职 (创始人股权 35%, 锁定 36 月)"
  ],
  "reasoning_chain": [
    {
      "step_no": 1, "title": "业务质量评估",
      "premise": "公司主业为协作机器人, 第二曲线为具身智能",
      "data_source": "prospectus_analyst: 营收 4.2 亿 (CAGR 35%), 毛利 38%",
      "calculation": "对比行业中位 (营收 2 亿, 毛利 32%): 营收高 100%, 毛利高 6pp",
      "conclusion": "业务质量优于行业中位, 给基础估值 + 10% 溢价",
      "confidence": "中",
      "caveats": ["客户集中度 top5 = 65% 偏高, 单大客户失约影响显著"]
    },
    {
      "step_no": 2, "title": "估值起点 — 可比公司 PS 锚定",
      "premise": "采用 PS 估值, 因公司尚未稳定盈利",
      "data_source": "comparable: 越疆 PS=27 / 华沿 PS=18, 中位 22x",
      "calculation": "22 × 3.6 亿 RMB = 79.2 亿 RMB ≈ 88 亿 HKD (汇率 1.1)",
      "conclusion": "PS 法基准估值 ≈ 88 亿 HKD",
      "confidence": "中",
      "caveats": ["可比仅 2 家, 样本小; 越疆已上市 1 年, 估值更稳定"]
    },
    "... (再列 5-8 步)"
  ],
  "decision_weights": [
    {
      "factor": "业务质量", "weight": 0.20, "score": 4.0,
      "source_agents": ["prospectus_analyst", "tech_trend"],
      "rationale": "权重 20% 取标准区间下沿——成长股但客户集中度风险偏高, 不宜过度溢价业务质量; 得分 4.0 因为营收 CAGR 35% 显著高于行业, 毛利 38% 高于中位 35%"
    },
    {
      "factor": "估值合理性", "weight": 0.30, "score": 3.0,
      "source_agents": ["comparable"],
      "rationale": "权重 30% 取标准区间中段——港股 IPO 破发率高, 估值是首要约束; 得分 3.0 因为 PS 22x 处于可比中位附近, 不偏低也不极端高估"
    },
    {
      "factor": "风控等级", "weight": 0.25, "score": 3.5,
      "source_agents": ["risk"],
      "rationale": "权重 25% 标准——风控独立, 否决权大; 得分 3.5 因为风控综合评级 3.5/5, 主要风险点 (客户集中) 已识别且有 kill switch"
    },
    {
      "factor": "宏观与行业窗口", "weight": 0.10, "score": 4.0,
      "source_agents": ["macro", "industry"],
      "rationale": "权重 10% 标准下沿——宏观影响时点不影响项目本身; 得分 4.0 因为 macro Agent 给窗口期评分 4/5, 行业景气度高"
    },
    {
      "factor": "辩论倾向", "weight": 0.10, "score": 3.5,
      "source_agents": ["debate_manager"],
      "rationale": "权重 10% 标准下沿——辩论是双向校验, 不是独立信号; 得分 3.5 因为 debate_manager 裁决倾向 Bull, 但 Bear 提出的客户集中风险有数据支持"
    },
    {
      "factor": "情绪与流动性", "weight": 0.05, "score": 3.5,
      "source_agents": ["sentiment"],
      "rationale": "权重 5% 标准下沿——短期情绪对锁定期影响有限; 得分 3.5 因为同期机器人板块情绪中性偏积极, 但同行业同期 IPO 较多, 资金分流风险存在"
    }
  ]
}
```

注: 上述示例中 weight 之和 = 1.00, 加权总分 = 0.20×4.0 + 0.30×3.0 + 0.25×3.5 +
0.10×4.0 + 0.10×3.5 + 0.05×3.5 = 0.80 + 0.90 + 0.875 + 0.40 + 0.35 + 0.175
= 3.50 → 落在 3.0-4.0 区间 → "审慎参与", 与最终 recommendation 一致.

之后用以下 Markdown 章节展开论述：

## 一、投决论述
综合论证，引用各 Agent 简报中的证据。这一章是给"快速读者"的——执行摘要后的扩展版。

## 二、关键假设清单 (Key Assumptions) ⭐
列出本投决依赖的 5-10 条核心假设, 每条标注:
- 假设内容（可量化）
- 数据来源（哪个 Agent / 招股书第几页）
- 假设被颠覆的后果（决议如何变化）
**这一章让用户看到"如果哪条假设错了, 结论会不同"**

## 三、估值方法分拆详解 ⭐
对每种估值方法（PE / PS / PEG / SOTP / DCF）分别展开:
- 为什么选这个方法（公司特征 / 行业惯例）
- 输入数据（peer / target metric, 来自哪个 Agent）
- 计算公式 + 中间结果
- 该方法的局限性
**这一章让用户看到"22x × 3.6 = 79"这种具体推算, 而非"凭经验估 80 亿"**

## 四、敏感性分析详解 ⭐
针对悲观/基准/乐观三档, 解释:
- triggers 为何可量化、阈值如何设定
- valuation 如何从 trigger 反推（公式或文字描述）
- probability 如何主观赋值（历史基准 / 专家判断 / 蒙特卡洛）

## 五、推理链 (Reasoning Chain) ⭐⭐
**这是本份 memo 的核心交付物之一**——把"从一堆 Agent 简报推到最终建议"的链条显式化:
- 5-10 步关键推理, 每步标注 前提 → 数据 → 计算 → 结论 → 异议
- 让用户能 trace 任意量化结论的来源

## 六、决策因子加权打分 ⭐⭐⭐
**这是本份 memo 的另一核心交付物**——把"多 Agent 简报综合到最终建议"的过程从黑箱
变为白箱: 5-7 个因子各自独立打分 + 显式权重 → 加权总分 → 推荐档位映射.
- 每个因子的 weight 必须有 rationale (说明为什么这个权重)
- 每个因子的 score 必须引用具体 Agent 简报 (说明为什么这个分)
- 加权总分对应的推荐区间必须与最终 recommendation 一致 (否则需在论述里说明
  "为什么我覆盖了加权结果")

## 七、对冲与退出策略
解释 hedging_strategy 和 exit_plan 的逻辑, 为什么这种节奏 / 工具。

## 八、Kill Switches 触发条件
解释每条退出触发的合理性, 指出对应的 monitoring_kpis 和阈值依据。

## 九、与 Bull/Bear 辩论的关系
说明你采纳了哪一方哪些观点, 为什么。

## 十、风险敞口与不确定性
列出主要不确定性和应对方式, 引用风控简报。"""


_REC_VALUES = {"认购", "审慎参与", "观望", "不认购"}
_CONF_VALUES = {"高", "中", "低"}
_PRICING_VALUES = {"估值偏低", "合理", "偏高", "严重高估"}


class ValuationMethodResult(BaseModel):
    """单一估值方法的具体推算（让用户看到 22x × 3.6 = 79）。"""
    model_config = ConfigDict(extra="ignore")
    method: str  # "PE" / "PS" / "PEG" / "EV/EBITDA" / "SOTP" / "DCF"
    peer_basis: str = ""  # 例: "可比公司 PS 中位数 22x (越疆 27x / 华沿 18x)"
    target_metric: str = ""  # 例: "2025E 营收 3.6 亿 RMB"
    formula: str = ""  # 例: "22 × 3.6 = 79.2 亿 RMB ≈ 88 亿 HKD"
    result_hkd_b: float | None = None
    weight: float = Field(default=1.0, ge=0, le=1, description="该方法在加权综合估值中的权重")


class ValuationRange(BaseModel):
    low: float | None = None
    mid: float
    high: float | None = None
    anchor_method: str
    anchor_logic: str
    # 新增: 估值推算的具体分拆（让用户看到每种方法的"输入 → 公式 → 输出"）
    methodology_breakdown: list[ValuationMethodResult] = Field(
        default_factory=list,
        description="每种估值方法的具体推算（PE/PS/PEG/SOTP/...）",
    )
    key_assumptions: list[str] = Field(
        default_factory=list,
        description="估值依赖的 3-5 条关键假设（如:营收 CAGR 35%、毛利稳定 38%）",
    )


# --- 专业级新增：敏感性 / 对冲 / 退出 / kill switches / 详细 KPI ---

class ScenarioRow(BaseModel):
    """敏感性分析的一行情景。"""
    model_config = ConfigDict(extra="ignore")
    name: str  # 悲观 / 基准 / 乐观（或更细分）
    triggers: list[str] = Field(default_factory=list, description="触发该情景的 1-3 条可量化条件")
    valuation_hkd_b: float
    probability: float = Field(ge=0, le=1)
    expected_return_pct: float | None = Field(default=None, description="该情景下相对招股价回报率")
    # 新增: 推理过程（让用户看到为什么是这个估值/概率）
    valuation_derivation: str = Field(
        default="",
        description="估值推算路径，例：'毛利率从 38% 降至 30% → PS 压缩 25% → 80×0.75=60'",
    )
    probability_rationale: str = Field(
        default="",
        description="概率赋值依据，例：'近 3 月港股机器人 IPO 6 个月内破发率 30%'",
    )


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
    # 新增: 阈值设定的依据
    rationale: str = Field(
        default="",
        description="为什么这个阈值（行业基准/历史经验/风控保守度）",
    )
    historical_precedent: str = Field(
        default="",
        description="可选:相似案例（如:'2024 年 X 公司 CTO 离职 30 天股价跌 40%'）",
    )


class MonitoringKPI(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    threshold: str = ""
    frequency: str = "季报"
    action_if_breach: str = ""
    # 新增: KPI 阈值的设定依据 + 行业基准
    threshold_rationale: str = Field(
        default="",
        description="为什么阈值是这个数（行业中位数/历史 1.5σ/招股书披露的承诺值）",
    )
    industry_benchmark: str = Field(
        default="",
        description="可选:行业基准对比（如：行业中位数 = 35%, 阈值 30% 留 5pp 缓冲）",
    )


# ============================================================================
# 决策因子加权打分卡 (v4 新增)
#
# 真投行 IC memo 处理"如何把多 Agent 简报综合到最终建议"的方法不是"加权求和"
# (投决本质是 multi-factor judgment, 不是 scoring formula), 而是:
# 1. 因子打分卡: 每维度独立 1-5 分 + 显式权重
# 2. 加权总分 → 推荐档位映射
# 3. 否决条件 (veto): 任一硬条件不满足直接否决, 不论加权分多高
#
# 我们的设计:
# - reasoning_chain 解释"逻辑推导链"(前提→数据→计算→结论, 链状)
# - decision_weights 解释"加权打分卡"(各维度独立打分 + 权重, 求和)
# 两者互补: 推理链是定性论证, 打分卡是定量综合.
# ============================================================================

# 推荐的标准因子集合 + 权重区间 (LLM 可根据项目特征调整)
STANDARD_DECISION_FACTORS: list[dict[str, Any]] = [
    {
        "factor": "业务质量",
        "suggested_weight_range": (0.15, 0.30),
        "source_agents": ["prospectus_analyst", "tech_trend"],
        "guidance": "基本盘业务质量 + 第二曲线兑现概率 + 技术壁垒. "
                    "成长股可上浮至 0.30; 困境反转/周期股下浮至 0.15.",
    },
    {
        "factor": "估值合理性",
        "suggested_weight_range": (0.20, 0.35),
        "source_agents": ["comparable", "scarcity"],
        "guidance": "港股 IPO 破发率高, 估值是首要约束. 估值偏高时该因子分数应低; "
                    "估值偏低时分数高. 即使其他因子优秀, 估值过高仍应限制权重. "
                    "**ScarcityAgent 的稀缺度评分作为估值合理性的修正项**: "
                    "scarcity_score ≥ 4 (稀缺) 允许 PS 中位 + 10-20% 溢价; "
                    "scarcity_score ≤ 2 (拥挤) 应在 PS 中位上压 10%+ 折扣.",
    },
    {
        "factor": "风控等级",
        "suggested_weight_range": (0.20, 0.30),
        "source_agents": ["risk"],
        "guidance": "风控 8 维评级综合得分 (1-5, 5=极低风险). 否决条件由风控决定, "
                    "权重必须高. 高客户集中度/财务造假嫌疑/治理结构问题等需上调.",
    },
    {
        "factor": "宏观与行业窗口",
        "suggested_weight_range": (0.05, 0.20),
        "source_agents": ["macro", "industry"],
        "guidance": "影响时点不影响项目本身. 牛市/赛道热点上浮; 熊市/赛道遇冷下浮. "
                    "这个因子分数低时应降低认购金额而非否决.",
    },
    {
        "factor": "辩论倾向",
        "suggested_weight_range": (0.10, 0.20),
        "source_agents": ["debate_manager"],
        "guidance": "Bull/Bear 辩论裁决 + 事后可验证清单. 是双向校验维度, 不是独立信号. "
                    "辩论分歧大时该维度本身分数中性, 但权重可调高强调'信息缺口'.",
    },
    {
        "factor": "情绪与流动性",
        "suggested_weight_range": (0.05, 0.15),
        "source_agents": ["sentiment"],
        "guidance": "市场情绪 + 路演渠道信号 + 同期 IPO 资金分流. 短期影响打新效果, "
                    "对 6 月禁售期内的破发概率有直接关联.",
    },
]


class FactorWeight(BaseModel):
    """单一决策因子的权重 + 得分。

    最终建议 = 各因子的 weight × score 加权求和, 映射到 推荐档位.
    """
    model_config = ConfigDict(extra="ignore")
    factor: str = Field(description="因子名, 例: '业务质量' / '估值合理性' / '风控等级'")
    weight: float = Field(ge=0, le=1, description="该因子在决议中的权重, 各因子之和应 ≈ 1.0")
    score: float = Field(ge=0, le=5, description="该因子当前得分, 0=最差 5=最优")
    contribution: float = Field(default=0.0, description="weight × score, 缺失时自动算")
    source_agents: list[str] = Field(
        default_factory=list,
        description="该因子打分依据的 Agent 名 (用于 trace 来源)",
    )
    rationale: str = Field(
        default="",
        description="必填: 为什么给这个权重 + 为什么这个得分",
    )

    @model_validator(mode="after")
    def _ensure_contribution(self) -> "FactorWeight":
        # contribution 优先取 LLM 输出, 缺失或异常时自动算 weight × score
        if self.contribution == 0.0 or self.contribution > 5.0:
            object.__setattr__(self, "contribution", round(self.weight * self.score, 3))
        return self


# 加权总分 → 推荐档位映射 (各档区间)
RECOMMENDATION_MAPPING: list[tuple[float, float, str]] = [
    (4.0, 5.0, "认购"),
    (3.0, 4.0, "审慎参与"),
    (2.0, 3.0, "观望"),
    (0.0, 2.0, "不认购"),
]


def map_score_to_recommendation(score: float) -> str:
    """加权总分 → 推荐档位."""
    for low, high, rec in RECOMMENDATION_MAPPING:
        if low <= score < high:
            return rec
    if score >= 5.0:
        return "认购"
    return "不认购"


def format_score_mapping(score: float) -> str:
    """渲染映射区间说明, 例: '3.50 → 审慎参与 (3.0-4.0 区间)'."""
    rec = map_score_to_recommendation(score)
    for low, high, r in RECOMMENDATION_MAPPING:
        if r == rec:
            return f"{score:.2f} → {rec} ({low:.1f}-{high:.1f} 区间)"
    return f"{score:.2f} → {rec}"


class ReasoningStep(BaseModel):
    """投决推理链的一步。

    一份顶级 IC memo 的核心是"从假设到结论的论证流"，5-10 步链式展示让用户能 trace
    每个量化结论的来源。
    """
    model_config = ConfigDict(extra="ignore")
    step_no: int = 1  # 步骤序号
    title: str  # 例: "估值起点：从可比公司 PS 中位数推出基准估值"
    premise: str = ""  # 该步骤的前提条件
    data_source: str = ""  # 数据出处 (引用 Agent 名 + 具体数字)
    calculation: str = ""  # 计算 / 推算过程 (公式或文字)
    conclusion: str = ""  # 该步骤的结论
    confidence: Literal["高", "中", "低"] = "中"
    caveats: list[str] = Field(default_factory=list, description="该步骤的潜在异议或假设依赖")


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

    # 推理链 v3 新增（让用户看到每个量化结论的"为什么"）
    key_assumptions: list[str] = Field(
        default_factory=list,
        description="本投决依赖的 5-10 条核心假设（如:营收 CAGR 35% / 毛利稳定 38% / "
        "行业 PE 不发生系统性压缩）。如果其中任一被颠覆, 决议需重新评估。",
    )
    reasoning_chain: list[ReasoningStep] = Field(
        default_factory=list,
        description="5-10 步关键推理（前提→数据→计算→结论），让用户从最终建议 trace "
        "回每个量化指标的来源",
    )

    # 决策因子加权打分 v4 新增（显式权重, 让用户看到每个维度的贡献）
    decision_weights: list[FactorWeight] = Field(
        default_factory=list,
        description="决策因子加权打分卡 (5-7 个因子). 各 weight 之和应 ≈ 1.0; "
        "加权总分 = Σ(weight × score), 映射到推荐档位 (≥4 认购 / 3-4 审慎 / "
        "2-3 观望 / <2 不认购).",
    )
    weighted_total_score: float | None = Field(
        default=None,
        description="加权总分 0-5, 缺失时由 model_validator 自动算",
    )
    weighted_to_recommendation_mapping: str = Field(
        default="",
        description="例: '3.50 → 审慎参与 (3.0-4.0 区间)'. 缺失时自动渲染",
    )

    @model_validator(mode="after")
    def _ensure_weighted_total(self) -> "DecisionResult":
        """自动算加权总分 + 映射文本 + 校验权重之和。"""
        if not self.decision_weights:
            return self
        # 自动算 contribution 之和
        if self.weighted_total_score is None:
            object.__setattr__(
                self,
                "weighted_total_score",
                round(sum(f.contribution for f in self.decision_weights), 3),
            )
        # 自动渲染映射文本
        if not self.weighted_to_recommendation_mapping and self.weighted_total_score is not None:
            object.__setattr__(
                self,
                "weighted_to_recommendation_mapping",
                format_score_mapping(self.weighted_total_score),
            )
        # 软警告: 权重之和偏离 1.0
        total_w = sum(f.weight for f in self.decision_weights)
        if not (0.92 <= total_w <= 1.08):
            logger.warning(
                f"[Decision] 决策因子权重之和 = {total_w:.3f}, 偏离 1.0 (±8%); "
                f"加权总分可能不可比. 因子: {[f.factor for f in self.decision_weights]}"
            )
        return self


def _render_weight_priors_for_prompt(priors: dict | None) -> str:
    """渲染历史权重校准 priors 为 Decision Agent 的 prompt 输入。

    priors 来自 store.get_weight_calibration_priors(): 同行业历史 closed 项目的
    PostmortemAgent 输出的 weight_calibrations 聚合.

    样本不足或无 calibration 时返回空字符串 (不污染 prompt)。
    """
    if not priors:
        return ""
    n = priors.get("sample_size", 0)
    cals = priors.get("calibrations") or []
    min_required = priors.get("min_samples_required")
    if not cals:
        # 样本不足或无 calibration, 显式说明 (LLM 看到后知道是冷启动期)
        if n > 0 and min_required:
            return (
                f"# 历史权重校准参考\n"
                f"（同行业历史 closed 项目仅 {n} 个 < 最低门槛 {min_required}, "
                f"暂无聚合校准, 按 STANDARD_DECISION_FACTORS 区间自定）\n\n"
            )
        return ""
    # 渲染表格
    lines = [
        "# 历史权重校准参考 ⭐",
        f"基于同行业 {n} 个已闭环的历史项目（PostmortemAgent 复盘后的"
        f"权重调整建议聚合）, 以下因子在事后看通常被错误估权重. "
        f"**这是软引导, 不是硬约束**——请参考但仍按本项目特征自定权重, 在 rationale "
        f"中说明你为何采纳/偏离这些 prior:",
        "",
        "| 因子 | 历史平均偏差 (suggested - actual) | 样本数 | 偏差区间 | 典型理由 |",
        "|---|---|---|---|---|",
    ]
    for c in cals[:10]:  # 最多展示 10 个
        delta = c.get("avg_delta", 0)
        delta_str = f"{delta:+.2%}" if delta else "0%"
        n_samples = c.get("samples", 0)
        rng = f"[{c.get('min_delta', 0):+.2%}, {c.get('max_delta', 0):+.2%}]"
        rationale_examples = c.get("rationale_examples", []) or []
        rationale_str = (rationale_examples[0][:80] + "…") if rationale_examples else "—"
        direction = "应**上调**" if delta > 0 else ("应**下调**" if delta < 0 else "保持")
        lines.append(
            f"| {c.get('factor', '—')} | {delta_str} ({direction}) | "
            f"{n_samples} | {rng} | {rationale_str} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


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


def validate_decision(
    parsed: dict[str, Any] | None,
    *,
    listing_profile: Any = None,
    extra_risk_dim_names: list[str] | None = None,
    strict: bool = True,
) -> tuple[DecisionResult | None, str]:
    """校验 LLM 输出的决议 JSON.

    Args:
        parsed: LLM 输出 JSON
        listing_profile: 当前项目 ListingProfile (用于估值方法禁用清单 / 额外风险维度强制覆盖)
        extra_risk_dim_names: 必须覆盖的风险维度名 (来自 listing_profile.extra_risk_dimensions)
        strict: True (默认) 时硬约束触发返 None 让 retry; False 时仅 warning

    Returns: (model, error_message)；error_message 非空表示校验失败 (retry 时用)。
    """
    if parsed is None:
        return None, "无法从输出中提取 ```json``` 代码块。"
    try:
        result = DecisionResult.model_validate(parsed)
    except ValidationError as e:
        return None, str(e)

    warnings: list[str] = []
    hard_errors: list[str] = []

    # ---------- 软校验: 枚举值 ----------
    if result.recommendation not in _REC_VALUES:
        warnings.append(f"recommendation 应为 {_REC_VALUES}，得到 '{result.recommendation}'")
    if result.confidence not in _CONF_VALUES:
        warnings.append(f"confidence 应为 {_CONF_VALUES}，得到 '{result.confidence}'")
    if result.ipo_pricing_view not in _PRICING_VALUES:
        warnings.append(f"ipo_pricing_view 应为 {_PRICING_VALUES}，得到 '{result.ipo_pricing_view}'")

    # ---------- P1.4 硬约束: weighted_total → recommendation 一致性 ----------
    if (
        result.decision_weights
        and result.weighted_total_score is not None
        and result.recommendation in _REC_VALUES
    ):
        expected_rec = map_score_to_recommendation(result.weighted_total_score)
        if expected_rec != result.recommendation:
            hard_errors.append(
                f"加权总分 {result.weighted_total_score:.2f} 映射到 '{expected_rec}', "
                f"但 recommendation 给的是 '{result.recommendation}'。两者必须一致 — "
                f"要么调整加权打分让总分落在 '{result.recommendation}' 区间, "
                f"要么把 recommendation 改为 '{expected_rec}'。"
            )

    # ---------- P1.5 硬约束: 估值方法禁用清单 ----------
    if listing_profile is not None:
        try:
            from src.tools.valuation_engine import select_methods_by_profile
            sel = select_methods_by_profile(listing_profile)
            forbidden = sel.get("forbidden", set())
            used_methods = {
                m.method.upper().strip()
                for m in result.valuation_range_hkd_billion.methodology_breakdown
                if m.method
            }
            forbidden_upper = {x.upper().strip() for x in forbidden}
            violations = used_methods & forbidden_upper
            if violations:
                hard_errors.append(
                    f"上市档案 ({getattr(listing_profile, 'listing_chapter', '')}) 禁用方法 "
                    f"{forbidden}, 但 methodology_breakdown 使用了 {violations}。"
                    f"请改用主用方法 {sel.get('primary', set())}。"
                )
        except Exception:
            pass  # ListingProfile 解析异常时不阻塞

    # ---------- P1.6 硬约束: reasoning_chain 最少 5 步 ----------
    if result.reasoning_chain and len(result.reasoning_chain) < 5:
        hard_errors.append(
            f"reasoning_chain 仅 {len(result.reasoning_chain)} 步, 必须至少 5 步 "
            f"(覆盖业务质量/估值起点/风险约束/宏观窗口/最终建议等关键推理点)。"
        )

    # ---------- P1.7 硬约束: risk 覆盖 ListingProfile 的 extra_risks ----------
    if extra_risk_dim_names:
        # 收集 LLM 给出的风险维度名
        llm_dims_text = " ".join(result.key_risks)  # key_risks 是文本列表
        # 也检查 reasoning_chain 中是否提及
        for step in result.reasoning_chain:
            llm_dims_text += " " + step.title + " " + step.conclusion
        missing = []
        for dim in extra_risk_dim_names:
            # 取核心关键字 (去掉"风险"等通用后缀) 做模糊匹配
            core_kw = dim.replace("风险", "").replace("评估", "").strip()[:6]
            if core_kw and core_kw not in llm_dims_text:
                missing.append(dim)
        if missing:
            hard_errors.append(
                f"上市档案要求覆盖 {len(extra_risk_dim_names)} 个特殊风险维度, "
                f"但 key_risks + reasoning_chain 未提及: {missing}。请显式增加。"
            )

    if hard_errors and strict:
        # 硬约束触发, 返 None 让 LLM retry
        return None, "硬约束违反:\n" + "\n".join(f"- {e}" for e in hard_errors)

    if hard_errors:
        warnings.extend(hard_errors)
    return result, "; ".join(warnings)


class DecisionAgent(BaseAgent):
    name = "decision"
    tier = ModelTier.DECIDE
    description = "最终基石投资决策 Agent"
    fatal = True  # 决策失败 = 没有产出，整个流程白跑

    def run(self, ctx: AgentContext) -> AgentReport:
        briefs_text = _briefs_block(ctx.briefs)
        weight_priors_block = _render_weight_priors_for_prompt(
            ctx.extras.weight_priors if hasattr(ctx.extras, "weight_priors") else {}
        )
        # ListingProfile 注入（决定调权 / 估值方法 / 风险维度的硬约束）
        profile_block = ""
        listing_profile = getattr(ctx.extras, "listing_profile", None)
        extra_risk_dim_names: list[str] = []
        try:
            from src.agents.listing_profile import (
                render_profile_for_prompt,
                extra_risk_dimensions,
            )
            profile_block = render_profile_for_prompt(listing_profile)
            if listing_profile is not None:
                extra_risk_dim_names = [
                    d.get("dimension", "") for d in extra_risk_dimensions(listing_profile)
                    if d.get("dimension")
                ]
        except Exception:
            pass

        user_msg = (
            f"# 待决策项目\n"
            f"- 公司：{ctx.company_name} ({ctx.ticker})\n"
            f"- 行业：{ctx.industry}\n"
            f"- 项目 ID：{ctx.project_id}\n\n"
            f"{profile_block}"
            f"{weight_priors_block}"
            f"# 各 Agent 简报\n\n{briefs_text}\n\n"
            f"请按系统指令的格式输出最终基石投资决策。\n\n"
            f"**重要**: 如果上方有 # 上市档案 块, 你的 decision_weights 权重必须在该档案"
            f"调整后的区间内, 估值方法必须采纳推荐主用方法 (禁用清单内的方法不能出现在"
            f"valuation_range_hkd_billion.methodology_breakdown), 风险评估必须覆盖"
            f"额外风险维度。"
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
        result, err = validate_decision(
            parsed,
            listing_profile=listing_profile,
            extra_risk_dim_names=extra_risk_dim_names or None,
        )

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
            result, err = validate_decision(
                parsed,
                listing_profile=listing_profile,
                extra_risk_dim_names=extra_risk_dim_names or None,
            )
            if result is None:
                logger.error(f"决议 JSON 重试仍失败: {err[:300]}")

        # SensitivityEngine 后处理：用引擎从 base case 重算三档情景, 作为可审计 anchor
        if result is not None:
            try:
                from src.tools.sensitivity_engine import (
                    BaseCase,
                    compute_sensitivity_table,
                )
                base_mid = result.valuation_range_hkd_billion.mid
                base_case = BaseCase(
                    valuation_hkd_b=float(base_mid),
                    valuation_at_ipo_hkd_b=float(base_mid),
                )
                listing_chapter = (
                    getattr(listing_profile, "listing_chapter", "Unknown")
                    if listing_profile else "Unknown"
                )
                size_tier = (
                    getattr(listing_profile, "size_tier", "Unknown")
                    if listing_profile else "Unknown"
                )
                engine_rows = compute_sensitivity_table(
                    base_case,
                    listing_chapter=listing_chapter,
                    size_tier=size_tier,
                )
                ctx.extras.misc["engine_sensitivity"] = [
                    {
                        "name": r.name,
                        "valuation_hkd_b": r.valuation_hkd_b,
                        "probability": r.probability,
                        "expected_return_pct": r.expected_return_pct,
                        "valuation_derivation": r.valuation_derivation,
                        "probability_rationale": r.probability_rationale,
                        "triggers": r.triggers,
                    }
                    for r in engine_rows
                ]
            except Exception as exc:
                logger.warning(f"[Decision] SensitivityEngine 后处理失败: {exc}")

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
