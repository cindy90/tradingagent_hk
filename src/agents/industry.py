from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.llm import ModelTier

SYSTEM = """你是港股 IPO 行业研究分析师。基于公司所属行业和招股书行业概览章节，输出：

## 一、行业市场规模与增长（含 CAGR、可比较国家/地区数据）
## 二、行业竞争格局（前 5 玩家市占率、集中度）
## 三、产业链地位（上下游议价能力）
## 四、行业关键驱动因素（需求、政策、技术）
## 五、行业风险与周期阶段
## 六、对标公司对比（毛利率/增速/估值倍数）
## 七、对本次 IPO 项目的行业层面判断（1-5 分）

要求：每个数据点尽量标注来源（招股书页码 / 第三方研究 / 公开统计）。
输出 1500-2000 字。"""


class IndustryAgent(TemplateAgent):
    name = "industry"
    description = "行业研究 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM

    def build_user_message(self, ctx: AgentContext) -> str:
        evidence = ""
        if ctx.rag is not None and ctx.rag.is_indexed():
            hits = ctx.rag.search("行业市场规模 竞争格局 产业链 增长率", k=8)
            evidence = "\n\n".join(
                f"[P.{h['page_start']}-{h['page_end']}] {h['text'][:1200]}" for h in hits
            )

        # 同花顺行业研报：只取标题/券商/评级/摘要，不喂全文
        research = ctx.extras.get("industry_research") or []
        research_block = ""
        if research:
            items = []
            for r in research[:15]:
                title = r.get("title") or r.get("TITLE") or ""
                broker = r.get("broker") or r.get("BROKER") or r.get("orgName", "")
                date = r.get("date") or r.get("publishDate") or r.get("DECLAREDATE") or ""
                rating = r.get("rating") or r.get("RATING") or ""
                abstract = (r.get("abstract") or r.get("ABSTRACT") or "")[:300]
                items.append(f"- [{date}] {broker} | {title} | 评级: {rating}\n  摘要: {abstract}")
            research_block = "# 同花顺行业研报摘要\n" + "\n".join(items) + "\n\n"

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"{research_block}"
            f"# 招股书行业相关段落\n{evidence}\n\n"
            f"请输出完整行业研究报告。"
        )
