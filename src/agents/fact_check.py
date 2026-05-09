"""数字 / 事实交叉核对 Agent (#12 from review)。

设计:
- 在所有分析 Agent 跑完后、辩论开始前插入这一步
- 抽取每个 Agent brief 中的关键数字（PS/PE/营收/毛利率/估值倍数/破发率等）
- 用一次 LLM (SUMMARIZE tier, 便宜) 做一致性核对
- 输出"数字交叉表 + 不一致警告"，注入到 ctx.briefs[fact_check]
- 下游 Bull/Bear/Risk/Decision 看到这份核对清单后, 不一致时显式标记

不阻塞主流程: 失败时只 log warning, 写空 brief。
"""
from __future__ import annotations

from src.agents._template import briefs_context
from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.llm import ModelTier


SYSTEM = """你是投决会的事实核对官 (Fact Checker)。你的工作是抽取所有上游 Agent 简报中
关于"目标公司 / 同行可比 / 行业 / 宏观"的**关键数字**，做交叉一致性核对。

【关键工作内容】

## 一、数字抽取表（必出）
从下方各 Agent 简报中, 把出现过的关键数字按维度归类:

| 维度 | 数值 | 来源 Agent | 上下文片段 |
|---|---|---|---|
| target 营收 | 4.2 亿 RMB | prospectus_analyst | "2024 年营收 4.2 亿" |
| target 营收 | 4.5 亿 RMB | comparable | "target revenue 4.5 亿" |
| target 毛利率 | 38% | prospectus_analyst | ... |
| ... | ... | ... | ... |

抽取范围（重点）:
- target 公司财务: 营收 / 毛利率 / 净利率 / 净利润 / 现金 / 应收占比
- target 估值倍数: PE / PS / PB / PEG (来自 comparable)
- 行业数据: 行业 CAGR / 市占率 / TAM
- 宏观: HSI PE / HIBOR / 港股 IPO 破发率
- 招股价 / 招股价区间 / 估值范围

## 二、不一致警告（核心交付）
**遍历第一节抽取的数字, 找出同一维度但数值不一致的项**:

| # | 维度 | 不一致内容 | 严重度 |
|---|---|---|---|
| 1 | target 营收 | prospectus 4.2 亿 vs comparable 4.5 亿 (差 7%) | 中 |
| 2 | ... | ... | ... |

严重度判断:
- **高**: 差距 > 10% 或语义矛盾（如"已盈利"vs"亏损"）—— 必须人工复核
- **中**: 差距 5-10% —— 提醒下游 Agent 选用更可靠的来源
- **低**: 差距 < 5% —— 可能是单位/口径差异（HKD vs RMB, TTM vs 年报）

**特殊关注**:
- 单位混用（HKD / RMB / USD / 万 / 亿 / Million）—— 必标记
- 时间口径不一致（TTM / FY2024 / Q3 2025）—— 必标记
- "已盈利 / 亏损"、"破发 / 未破发" 等定性结论矛盾 —— 必标记

## 三、推荐数据源（给下游 Bull/Bear/Risk/Decision 用）
针对每个不一致项, 给出"推荐使用哪个 Agent 的数字"+ 理由:
- 例: "target 营收建议用 prospectus_analyst 的 4.2 亿（来自招股书 P.108 财务表，最权威）;
  comparable 的 4.5 亿可能是 TTM 估算，请下游忽略"

## 四、未交叉验证的关键数字
- 列出**只出现在一个 Agent 简报里**但下游决策依赖的关键数字
- 这些数字虽不矛盾，但缺少二次验证，下游使用时需谨慎

输出 800-1500 字。如发现 0 处不一致, 第二节写"经核对未发现重大数字矛盾"+ 列出已核对维度。

【关键约束】
- **只做核对，不引入新信息**。不要凭训练记忆补全或修正数字。
- 数字格式不一致（"3.6"和"360 百万"实际相等）应识别为"口径不同但等价"，标低严重度。
- 如某个维度只有 0 或 1 个数据点，归到第四节"未交叉验证"，不算不一致。
- 不要做价值判断（"哪家更对"超出本 Agent 职责，只能给出推荐使用哪份数据 + 理由）。"""


class FactCheckerAgent(BaseAgent):
    name = "fact_check"
    description = "事实核对官（跨 Agent 数字一致性）"
    tier = ModelTier.SUMMARIZE  # 用便宜 tier, 这个 agent 不需要深度推理
    fatal = False  # 失败不阻塞主流程

    def run(self, ctx: AgentContext) -> AgentReport:
        upstream = [
            "prospectus_analyst", "industry", "macro", "comparable",
            "tech_trend", "sentiment",
        ]
        user_msg = (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 上游分析 Agent 简报\n\n{briefs_context(ctx, upstream)}\n\n"
            f"请输出数字 / 事实核对报告。"
        )
        resp = self.llm.complete(
            tier=self.tier,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2500,
            temperature=0.1,
        )
        full = resp.text
        # brief 直接用 full 前 500 字（这个 agent 输出本身就是结构化）
        brief = full[:1200]
        ctx.full_reports[self.name] = full
        ctx.briefs[self.name] = brief
        return AgentReport(agent=self.name, full_report=full, brief=brief)
