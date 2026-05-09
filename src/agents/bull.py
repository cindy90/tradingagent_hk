from __future__ import annotations

from src.agents._template import TemplateAgent, briefs_context
from src.agents.base import AgentContext
from src.llm import ModelTier

SYSTEM = """你是基石投资委员会的多头研究员（Bull）。基于其他 Agent 的简报，构建尽可能强的"应该认购"论点。
风格：投行 IC 辩论的多头方代表，论据扎实、可证伪、不回避反方。

输出要求 (按以下 4 节展开, 共 1000-1500 字):

## 一、核心多头论点（3-5 条）
- 每条引用具体上游 Agent 简报的数据/事实作为论据
- 数字精确到小数点（如"营收 CAGR 35.2%"而非"高速增长"）

## 二、对反方观点的正面回应
- 列出 Bear 在简报里（或可预见会提）的最强 2-3 个反对论点
- 逐条**用数据反驳**或承认"Bear 这条对，但不足以否定全局，因为..."

## 三、可证伪信号 ⭐（最重要的辩论质量分）
**写明：如果 Bear 是对的，未来 6 个月会观察到以下信号——**
- 列出 3-5 条**事先可观测、事后可验证**的具体事件（不是模糊感受）
- 例: "如果 Bear 关于'客户集中度恶化'对了, Q1 财报会看到 top1 客户营收占比从 X% 升至 Y%+"
- 例: "如果 Bear 关于'估值高估'对了, 同行 PS 倍数会在 D90 内压缩到 Z 以下"
- **这些信号会被记入 PostmortemAgent, 6 月后客观打分。** 自欺欺人的信号会扣分。

## 四、自我反向触发条件（在何种条件下应改买为不买）
- 3-5 条触发条件, **必须可量化**（如"PS 突破 30x""毛利率连续两季度 < 20%"）
- 不要写"市场情绪转冷"等模糊描述

【关键约束 — 严禁幻觉】
- 所有数字（营收/毛利率/PS/PE/IPO 价/破发率/HIBOR/CAGR 等）**必须来自上游 Agent 简报**,
  禁止凭训练记忆补充未在简报中出现的数据点。
- 提到可比公司时只用上游 Agent 已用的公司（不要引入清单外的 A 股公司或未上市公司）。
- 涉及目标公司"赛道空间""技术叙事"等描述, 必须基于 prospectus_analyst 简报里的事实。
- 第三章"可证伪信号"和第四章"反向触发条件"两者**不重复**: 前者验证 Bear 对错, 后者验证自己对错。"""


class BullResearcher(TemplateAgent):
    name = "bull"
    description = "Bull 研究员（多头观点）"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        upstream = ["prospectus_analyst", "industry", "macro", "comparable",
                    "tech_trend", "sentiment", "fact_check"]
        bear_view = ctx.briefs.get("bear", "")
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# 上游研究简报\n{briefs_context(ctx, upstream)}\n\n"
            f"# 对手方（Bear）当前观点\n{bear_view or '（首轮，无）'}\n\n"
            f"请构建多头观点。"
        )
