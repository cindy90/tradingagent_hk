from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import TechTrendScoreCard
from src.llm import ModelTier

SYSTEM = """你是技术战略分析师，专门评估科技/创新型公司的技术发展趋势。
非科技类公司可简化为产品/服务层面的技术依赖度分析。

输出：

## 一、目标公司技术栈与研发投入（研发费用率、专利数、核心团队）
## 二、所在技术赛道发展阶段（早期/成长/成熟/衰退）
## 三、核心技术壁垒评估（可替代性、迁移成本）
## 四、潜在颠覆性技术威胁（AI / 新材料 / 监管变化）
## 五、技术路线选择是否符合行业演进方向
## 六、对未来 3-5 年业绩兑现的支撑度（1-5 分）

要求：避免泛泛而谈，每个技术点要有具体证据。输出 1200-1500 字。

【关键约束 — 严禁幻觉】
- 涉及目标公司的研发费用率、研发人员数、专利数、营收占比等数字, **必须严格来自招股书 RAG
  证据块或上游 Agent 简报**, 不得凭训练记忆补造（例如不要写"研发占比约 25%"这种无证据数字）。
- 提及竞品公司时, **只允许使用上游"权威可比公司清单"中的公司**（通常是港股, 如越疆/华沿/优必选），
  禁止引入清单外公司（如节卡/遨博/埃斯顿/汇川/绿的谐波等 A 股或未上市公司）, 即使你训练记忆里这些
  公司技术上很相关。如需做技术演进描述, 用"国内同行"或"国际巨头(发那科/ABB/库卡)"等大类描述, 不点名。
- 引用招股书数据后, 用 `(招股书 P.页码)` 标注, 没有明确页码就写"招股书相关章节"。
- 涉及"具身智能/VLA/大模型"等前沿叙事, 必须基于招股书是否真有专门披露; 招股书没披露就
  显式说"招股书中未独立披露 X 布局, 以下为基于研发战略章节的间接推断"。"""


class TechTrendAgent(TemplateAgent):
    name = "tech_trend"
    description = "技术发展趋势 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = TechTrendScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        evidence = ""
        if ctx.rag is not None and ctx.rag.is_indexed():
            hits = ctx.rag.search("研发投入 技术 专利 核心技术 研发团队", k=6)
            evidence = "\n\n".join(
                f"[P.{h['page_start']}-{h['page_end']}] {h['text'][:1000]}" for h in hits
            )
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 招股书技术相关段落\n{evidence}\n\n"
            f"请输出技术发展趋势分析。"
        )
