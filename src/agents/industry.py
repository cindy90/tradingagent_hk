from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import IndustryScoreCard
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
输出 1500-2000 字。

【关键约束 — 严禁幻觉】
- 第六节"对标公司对比"**必须严格使用下方"权威可比公司清单"+ "可比公司财务+估值倍数"两个表里的真实数字**。
  禁止写"待补"——如果表里某字段值是 None / "—"，请显式说"该项数据缺失"，
  并就**已有字段**做实质对比（例如毛利率 / 净利率 / PS / PB 都有数据就用这些做对比）。
- 不得引入清单外的公司（特别是 A 股埃斯顿/汇川/绿的谐波等），即使你训练记忆里这些公司很相关。
- 清单为空（投决人选择 skip）时，第六节写"暂无确认的港股可比清单, 仅做行业格局描述"。
- 涉及目标公司（target）数据时, 不要从训练记忆补充, 只引用上方表格里有的数字。"""


def _fmt(v, digits=2, suffix=""):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_money(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f} 亿"
    if abs(v) >= 1e4:
        return f"{v/1e4:.2f} 万"
    return f"{v:.0f}"


def _render_peers_md(peers: list[dict]) -> str:
    """权威可比公司 IPO 信息表（招股价/上市日期/首日开盘）。"""
    if not peers:
        return "（投决人未确认可比公司清单，第六节请仅做行业格局定性描述）"
    lines = ["| 代码 | 公司 | 上市日期 | 招股价 | 招股价上限 | 首日开盘价 | 首日开盘涨跌% |",
             "|---|---|---|---|---|---|---|"]
    for p in peers:
        fdor = p.get("first_day_open_return")
        lines.append(
            f"| {p.get('thscode') or p.get('ticker','')} | {p.get('name','')} | "
            f"{p.get('ipo_date','-')} | {p.get('ipo_price','-')} | "
            f"{p.get('ipo_max_price','-')} | {p.get('first_day_open','-')} | "
            f"{('+' + str(fdor)) if (fdor is not None and fdor >= 0) else (str(fdor) if fdor is not None else '-')} |"
        )
    return "\n".join(lines)


def _render_financials_md(target: dict | None, peers: list[dict]) -> str:
    """target + peers 财务+估值倍数对比表（毛利率/净利率/ROE/PS/PB/EV-Sales）。"""
    rows: list[dict] = []
    if target:
        rows.append({**target, "_role": "**target**"})
    rows.extend({**p, "_role": "peer"} for p in peers)
    if not rows:
        return "（无数据）"
    yr = (target or peers[0] if peers else {}).get("year", "")
    lines = [
        f"| 角色 | 代码 | 公司 | {yr} 营收 | {yr} 净利 | 毛利率 | 净利率 | ROE | PS-TTM | PB | EV/Sales |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = r.get("name") or "（IPO 询价中）"
        lines.append(
            f"| {r.get('_role')} | {r.get('thscode') or r.get('ticker','')} | {name} | "
            f"{_fmt_money(r.get('revenue'))} | {_fmt_money(r.get('net_profit'))} | "
            f"{_fmt(r.get('gross_margin'), suffix='%')} | "
            f"{_fmt(r.get('net_margin'), suffix='%')} | "
            f"{_fmt(r.get('roe'), suffix='%')} | "
            f"{_fmt(r.get('ps_ttm'))} | {_fmt(r.get('pb_latest'))} | "
            f"{_fmt(r.get('ev_sales'))} |"
        )
    return "\n".join(lines)


class IndustryAgent(TemplateAgent):
    name = "industry"
    description = "行业研究 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = IndustryScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        evidence = ""
        if ctx.rag is not None and ctx.rag.is_indexed():
            hits = ctx.rag.search("行业市场规模 竞争格局 产业链 增长率", k=8)
            evidence = "\n\n".join(
                f"[P.{h['page_start']}-{h['page_end']}] {h['text'][:1200]}" for h in hits
            )

        # 同花顺行业研报：只取标题/券商/评级/摘要，不喂全文
        research = ctx.extras.industry_research
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

        peers_md = _render_peers_md(ctx.extras.peers)
        financials_md = _render_financials_md(
            ctx.extras.target_valuation, ctx.extras.peers
        )

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 权威可比公司清单（PeerSuggester + 投决人确认）\n{peers_md}\n\n"
            f"# target + 可比公司财务 + 估值倍数（来自 iFinD THS_BD, 最新年报+最新交易日）\n"
            f"{financials_md}\n\n"
            f"{research_block}"
            f"# 招股书行业相关段落\n{evidence}\n\n"
            f"请输出完整行业研究报告。"
            f"第六节'对标公司对比'**必须使用上方表格中的真实数字**, 不要写'待补'。"
        )
