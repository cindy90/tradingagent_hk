"""招股书深度分析 Agent（完整实现，作为模板参考）。

工作流程:
1. 用 RAG 按主题检索招股书相关段落（业务/财务/风险/募资用途/基石条款）；
2. 把检索到的段落作为 user message，让 LLM 生成结构化分析；
3. 招股书核心摘要章节通过 cached_system_blocks 缓存复用（多个 Agent 共享）；
4. 调用 Summarizer 生成 brief。

子主题列表是显式的（而不是让 LLM 自己决定问什么），这样 token 可控。
"""
from __future__ import annotations

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.agents.scoring import parse_score_card, schema_instruction, strip_score_card_block
from src.agents.summarizer import Summarizer
from src.feedback.models import ProspectusScoreCard
from src.llm import ModelTier

# 显式定义的检索主题。新增主题就加一行，不要让 LLM 自己发散查询。
RETRIEVAL_TOPICS: list[tuple[str, str]] = [
    ("business_model", "公司主营业务、收入结构、商业模式、客户构成"),
    ("competitive_advantage", "核心竞争力、技术壁垒、护城河、市场地位"),
    ("financial_performance", "近三年营收、毛利、净利润、现金流、关键财务指标"),
    ("risk_factors", "主要风险因素、监管风险、客户集中、供应链风险"),
    ("use_of_proceeds", "募集资金用途、扩产计划、研发投入"),
    ("cornerstone_terms", "基石投资者、禁售期、估值条款"),
    ("management_shareholders", "实际控制人、主要股东、管理层背景"),
    ("related_party", "关连交易、应收账款、关联方依赖"),
]

ANALYST_SYSTEM = """你是一名专门负责港股 IPO 基石投资尽调的资深分析师，有 10 年港股投行经验。
工作风格：依据导向、不臆测、对每个结论标注证据出处（招股书页码）。

你将收到：
- 公司基本信息
- 从招股书 RAG 中检索到的相关段落（已按主题分组，每段附页码）

输出要求：一份完整的招股书深度分析报告（Markdown），按以下章节组织：

## 一、业务模式与收入结构
## 二、核心竞争力与护城河
## 三、财务表现与质量
## 四、风险因素剖析
## 五、募集资金用途分析
## 六、基石投资条款解读
## 七、股权结构与管理层
## 八、关连交易与潜在问题
## 九、综合判断（5 项打分：业务、财务、风险、估值锚定基础、基石条款）

写作要求：
- 每个关键论断后用 `(招股书 P.页码)` 标注证据；
- 数字必须直接引用招股书原文，不要猜；
- 章节九给出 1-5 分打分（5 最高）和打分理由；
- 总长 1500-2500 字，不要堆砌套话。"""


class ProspectusAnalystAgent(BaseAgent):
    name = "prospectus_analyst"
    tier = ModelTier.ANALYZE
    description = "招股书深度分析 Agent"
    fatal = True  # 招股书分析失败时整个流程没意义，必须中止
    score_card_class = ProspectusScoreCard

    def __init__(self, llm, summarizer: Summarizer | None = None):
        super().__init__(llm)
        self.summarizer = summarizer or Summarizer(llm)

    def _gather_evidence(self, ctx: AgentContext) -> str:
        if ctx.rag is None or not ctx.rag.is_indexed():
            return "（招股书 RAG 索引不可用，请检查 PDF 是否已加载）"

        from config import get_settings
        k = get_settings().rag_top_k
        blocks: list[str] = []
        for topic_key, topic_query in RETRIEVAL_TOPICS:
            text_hits = ctx.rag.search(topic_query, k=k, type_filter="text")
            table_hits: list[dict] = []
            # 财务/可比/募资类主题额外取表格
            if topic_key in ("financial_performance", "use_of_proceeds"):
                table_hits = ctx.rag.search(topic_query, k=2, type_filter="table")
            hits = text_hits + table_hits
            if not hits:
                continue
            blocks.append(f"### 主题: {topic_key} ({topic_query})")
            for h in hits:
                tag = "📊 表格" if h.get("type") == "table" else ""
                blocks.append(
                    f"[招股书 P.{h['page_start']}-{h['page_end']} | "
                    f"{h['section'] or '无章节标题'} {tag}]\n"
                    f"{h['text'][:1500]}"
                )
        return "\n\n".join(blocks)

    def run(self, ctx: AgentContext) -> AgentReport:
        evidence = self._gather_evidence(ctx)
        basic = ctx.extras.company_basic
        basic_block = ""
        if basic:
            lines = [f"- {k}: {v}" for k, v in basic.items() if v not in (None, "", "--")]
            if lines:
                basic_block = "# 公司基础信息（同花顺）\n" + "\n".join(lines) + "\n\n"

        user_msg = (
            f"# 待分析公司\n"
            f"- 名称：{ctx.company_name}\n"
            f"- 拟上市代码：{ctx.ticker}\n"
            f"- 所属行业：{ctx.industry}\n\n"
            f"{basic_block}"
            f"# 招股书检索证据\n\n{evidence}\n\n"
            f"请基于以上证据生成完整的招股书深度分析报告。"
        )

        system = ANALYST_SYSTEM + schema_instruction(self.score_card_class)
        resp = self.llm.complete(
            tier=self.tier,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            cached_system_blocks=ctx.cached_blocks or None,
            max_tokens=6500,
            temperature=0.2,
        )
        raw = resp.text

        sc = parse_score_card(raw, self.score_card_class, agent_name=self.name)
        if sc is not None:
            cards = ctx.extras.misc.setdefault("score_cards", {})
            cards[self.name] = sc.model_dump()

        full = strip_score_card_block(raw)
        brief = self.summarizer.compress(full, agent_name=self.name)

        ctx.full_reports[self.name] = full
        ctx.briefs[self.name] = brief

        return AgentReport(agent=self.name, full_report=full, brief=brief)
