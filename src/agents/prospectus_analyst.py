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
# 设计原则: 既覆盖通用尽调维度 (business/financial/risk), 也召回"公司故事"相关
# 特定证据 (产品矩阵/第二增长曲线/AI 战略/出海), 防止 LLM 把上市核心叙事一笔带过。
RETRIEVAL_TOPICS: list[tuple[str, str]] = [
    ("business_model", "公司主营业务、收入结构、商业模式、客户构成"),
    ("product_lines", "产品矩阵、产品分类、收入构成、按产品线收入拆分、SKU"),
    ("growth_curve", "未来发展战略、第二增长曲线、新产品方向、增长驱动力、未来 3-5 年规划"),
    ("ai_embodied_intelligence",
     "具身智能 具身机械臂 AI 大模型 VLA 人形机器人 通用机器人 智能体"),
    ("overseas_strategy", "海外业务、全球化、出海、海外客户、海外收入占比、跨境业务"),
    ("competitive_advantage", "核心竞争力、技术壁垒、护城河、市场地位"),
    ("financial_performance", "近三年营收、毛利、净利润、现金流、关键财务指标"),
    ("risk_factors", "主要风险因素、监管风险、客户集中、供应链风险"),
    ("use_of_proceeds", "募集资金用途、扩产计划、研发投入"),
    ("cornerstone_terms", "基石投资者、禁售期、估值条款"),
    ("management_shareholders", "实际控制人、主要股东、管理层背景"),
    ("related_party", "关连交易、应收账款、关联方依赖"),
    # 新增: 承销团 + 红旗章节专用召回 (#7 #13)
    ("underwriting_team", "保荐人 全球协调人 账簿管理人 联席承销 承销商 承销团"),
    ("accounting_policy", "收入确认 应收账款 拨备 合同资产 研发资本化 会计政策 重要会计估计"),
]

ANALYST_SYSTEM = """你是一名专门负责港股 IPO 基石投资尽调的资深分析师，有 10 年港股投行经验。
工作风格：依据导向、不臆测、对每个结论标注证据出处（招股书页码）。

你将收到：
- 公司基本信息
- 从招股书 RAG 中检索到的相关段落（已按主题分组，每段附页码），主题包括 product_lines /
  growth_curve / ai_embodied_intelligence / overseas_strategy 等"上市叙事"相关分类

输出要求：一份完整的招股书深度分析报告（Markdown），按以下 12 章组织：

## 一、业务模式与收入结构（基本盘）
## 二、产品矩阵与增长曲线（基本盘 + 第二曲线 + 上市叙事）⭐
## 三、核心竞争力与护城河
## 四、财务表现与质量
## 五、风险因素剖析
## 六、募集资金用途分析
## 七、基石投资条款解读
## 八、股权结构与管理层
## 九、关连交易与潜在问题
## 十、承销团评估 ⭐
## 十一、红旗清单 — 招股书没说什么 ⭐
## 十二、综合判断（5 项打分：业务、财务、风险、估值锚定基础、基石条款）

【关键章节强制要求 — 第二章 "产品矩阵与增长曲线"】

这是港股 IPO 估值溢价的核心来源, 投行路演故事的灵魂。**必须深度展开**:

1. **产品矩阵拆解（基本盘）**: 按招股书披露的产品线分组（如硬件本体/软件平台/解决方案/服务），
   引用各产品线收入占比、毛利率差异、客户结构差异。如招股书有"按产品类别收入分类表"必须引用。

2. **第二增长曲线识别（重点！）**:
   - 检查招股书是否反复强调某个**新产品方向 / 新业务条线 / 新技术叙事**（如"具身智能/具身机械臂/
     AI/人形机器人/大模型 VLA"等关键词在招股书中频繁出现, 或募集资金有专项分配）。
   - **如果识别到, 必须独立成 1-2 段**展开:
     * 该方向的产品形态 / 商业化进度 / 竞品定位
     * 它在招股书中的地位（路演故事核心 / 募资专项 / 研发战略）
     * 对估值的潜在贡献（赛道空间 / 倍数溢价路径 / 实现概率）
   - **禁止把第二曲线一笔带过**或仅作为竞争力的一句子提及。

3. **上市叙事提炼**: 用 1-2 句话提炼公司路演的核心故事
   （例: "工业 + 协作机器人基本盘 + 具身机械臂第二曲线"）。

4. **海外/出海角度**（如有）: 海外收入占比变化, 全球化战略实质性进展。

【关键章节强制要求 — 第十章 "承销团评估"】

港股 IPO 保荐人质量直接影响定价合理性 + 后市表现. 必须含:

1. **保荐人列表**: 招股书"承销团"或"全球发售"章节披露的保荐人/全球协调人/账簿管理人
2. **保荐人质量定级**: 一线 (高盛/摩根士丹利/中金/招商证券国际) / 二线 / 其他
3. **历史业绩**: 如果上游 sentiment / 招股书披露了承销商近 12 月港股 IPO 表现, 引用首日破发率 /
   30 日表现等. 没有数据则显式说"承销商历史数据缺失, 仅做定性评估"
4. **推论**: 一线投行通常更克制（避免后市破发损失声誉), 二线投行常超额定价

【关键章节强制要求 — 第十一章 "红旗清单 — 招股书没说什么"】

招股书永远是 PR'd version，避开了什么往往比说了什么更重要. 这一章是顶级分析师与初级分析师
的分水岭. 必须含 4 类红旗（每类 2-5 条, 不必每类都有, 但至少覆盖 2 类）:

1. **会计政策疑点**:
   - 收入确认时点（"项目验收" vs "产品交付", 港股工程类公司常见)
   - 应收账款 / 合同资产 计提政策（拨备比例 vs 行业平均）
   - 研发资本化比例（资本化比例 > 30% 是红旗）
   - 一次性收入 / 非经常性损益占比

2. **数据缺口**（招股书该披露但没披露的）:
   - 例: "为什么不披露 top 10 客户名单 / 单一客户营收占比 > 5% 但匿名"
   - 例: "递延收入 / 合同负债余额未单列"
   - 例: "员工人均薪酬 / 销售费用率行业对比缺失"

3. **不一致点**:
   - 不同章节数字相互矛盾（例: 第三章说"99% 客户复购率"vs 第七章财务披露"前五大客户每年都换"）
   - 战略叙事与财务事实背离（例: 反复强调"全球化"但海外营收 < 5%）

4. **奇怪的关连交易 / 治理结构**:
   - 关连方金额合理但交易模式不符合商业逻辑
   - 突击入股 / 申报前股东重组 / 创始人股权质押异常
   - 双重股权 / 一票否决权 / 任何"创始人保留控制"安排

每条红旗格式: `[**红旗** 类型] 描述: 招股书 P.页码 → 风险解读: 这意味着...`
即使没找到红旗, 第十一章也要写"经审慎检查未发现明显红旗" + 列出已检查的维度清单（避免空话）。

写作要求：
- 每个关键论断后用 `(招股书 P.页码)` 标注证据；
- 数字必须直接引用招股书原文, 不要猜；
- 第十二章给出 1-5 分打分（5 最高）和打分理由；
- 总长 3500-5000 字（10 → 12 章后允许更长）, 但禁止堆砌套话。

【严禁偷懒】
- 第二章如果空泛或回避（"产品覆盖广泛""战略协同"等套话）= 报告失败。
- 第十一章如果只写"未发现红旗"而不列检查维度 = 偷懒。
- 即使招股书没有完整的第二曲线披露, 也必须显式说"招股书未独立披露 X 业务条线收入占比, 以下为
  基于募集资金用途/研发战略章节的间接推断"——保留诚实标记, 但不要回避问题。"""


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

        # 主题数从 8 增到 12 后, 单主题召回数下调到 4 (原默认 6), 控制总 token。
        # 总 chunks: 12 * 4 = 48 (与原 8 * 6 = 48 持平), 单 chunk 截断到 1000 字进一步压缩。
        k = 4
        blocks: list[str] = []
        for topic_key, topic_query in RETRIEVAL_TOPICS:
            text_hits = ctx.rag.search(topic_query, k=k, type_filter="text")
            table_hits: list[dict] = []
            # 财务/募资类主题额外取表格
            if topic_key in ("financial_performance", "use_of_proceeds", "product_lines"):
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
                    f"{h['text'][:1000]}"
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
            max_tokens=5000,
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
