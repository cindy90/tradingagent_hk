from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import ComparableScoreCard
from src.llm import ModelTier
from src.tools.valuation import comparable_valuation

SYSTEM = """你是港股 IPO 估值分析师，专门做可比公司估值。基于：
- 目标公司财务数据（营收/净利润/毛利率/PE-PS-PB 等估值倍数, 来自 iFinD 实时拉取）
- 可比公司同维度数据
- 可比公司近期二级市场行情

输出：

## 一、可比公司选择依据（行业、规模、商业模式契合度）
## 二、估值倍数对比表（PE / PS / PB / EV/EBITDA / EV/Sales）
## 三、目标公司估值区间（低 / 中 / 高三档，分别对应何种倍数水平）
## 四、与招股书招股价区间对比（如有）
## 五、估值锚定结论（建议基石可接受估值上限）

注意：
- 不要自己重新算数, 直接用下方表格里的数字；
- 港股 IPO 通常会比可比公司有 10-25% 的 IPO 折扣；
- 都亏损时 PE 不适用, 用 PS / EV-Sales / PB 做估值锚；
- 财务数据单位通常是 RMB（中国大陆公司在港上市），市值/股价是 HKD，对比 PS 时无需汇率换算（PS 倍数本身已经标准化）。

输出 1200-1500 字。

【关键约束 — 严禁幻觉】
- 估值倍数（PE/PS/PB/EV-EBITDA/EV-Sales）**必须严格使用下方"估值倍数表"中的数字**，
  不得编造"PS 8-10x"等过时数字（即使你训练记忆里有）。
- 是否破发的判断 = 当前股价 vs 招股价（招股价见"权威可比公司清单"）。
  例: 越疆 34.58 vs 18.8 → 未破发, 溢价 84%。禁止说"破发到 12"等矛盾说法。
- target 公司倍数若已在表中, 必须**直接对比**（如 target PS 70 vs peer PS 27 → 偏高 159%），
  推算"建议折扣后 PS 区间"（如 target 应给折扣 50% → PS 应在 30-40x）。
- 数据为 None / 缺失时显式说"该项数据缺失"，不要凭空补。"""


def _fmt_num(v, digits=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v, digits=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_money(v):
    """大数字: > 1 亿显示亿, > 万显示万, 否则原始。"""
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
    if not peers:
        return "（投决人未确认可比公司清单）"
    lines = ["| 代码 | 公司 | 上市日期 | 招股价 | 招股价上限 | 首日开盘价 | 首日开盘涨跌% |",
             "|---|---|---|---|---|---|---|"]
    for p in peers:
        fdor = p.get("first_day_open_return")
        lines.append(
            f"| {p.get('thscode') or p.get('ticker','')} | {p.get('name','')} | "
            f"{p.get('ipo_date','-')} | {p.get('ipo_price','-')} | "
            f"{p.get('ipo_max_price','-')} | "
            f"{p.get('first_day_open','-')} | "
            f"{('+' + str(fdor)) if (fdor is not None and fdor >= 0) else (str(fdor) if fdor is not None else '-')} |"
        )
    return "\n".join(lines)


def _render_valuation_md(target: dict | None, peers: list[dict]) -> str:
    """估值倍数表: target + peers 横向对比。"""
    rows: list[dict] = []
    if target:
        rows.append({**target, "_role": "target"})
    rows.extend({**p, "_role": "peer"} for p in peers)
    if not rows:
        return "（缺数据）"
    lines = ["| 角色 | 代码 | 公司 | PE-TTM | PB | PS-TTM | EV/EBITDA | EV/Sales |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        role = "**target**" if r.get("_role") == "target" else "peer"
        lines.append(
            f"| {role} | {r.get('thscode') or r.get('ticker','')} | "
            f"{r.get('name', '-')} | "
            f"{_fmt_num(r.get('pe_ttm'))} | {_fmt_num(r.get('pb_latest'))} | "
            f"{_fmt_num(r.get('ps_ttm'))} | {_fmt_num(r.get('ev_ebitda'))} | "
            f"{_fmt_num(r.get('ev_sales'))} |"
        )
    return "\n".join(lines)


def _render_fundamentals_md(target: dict | None, peers: list[dict]) -> str:
    rows: list[dict] = []
    if target:
        rows.append({**target, "_role": "target"})
    rows.extend({**p, "_role": "peer"} for p in peers)
    if not rows:
        return "（缺数据）"
    yr = (target or peers[0] if peers else {}).get("year", "")
    lines = [f"| 角色 | 代码 | 公司 | {yr} 营收 | {yr} 净利 | 毛利率 | 净利率 | ROE |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        role = "**target**" if r.get("_role") == "target" else "peer"
        lines.append(
            f"| {role} | {r.get('thscode') or r.get('ticker','')} | "
            f"{r.get('name', '-')} | "
            f"{_fmt_money(r.get('revenue'))} | {_fmt_money(r.get('net_profit'))} | "
            f"{_fmt_pct(r.get('gross_margin'))} | {_fmt_pct(r.get('net_margin'))} | "
            f"{_fmt_pct(r.get('roe'))} |"
        )
    return "\n".join(lines)


def _render_quotes_md(rows: list[dict]) -> str:
    if not rows:
        return "（未提供可比公司近期行情）"
    lines = ["| 代码 | 最新价 | 最新日期 | 30 日涨跌% | 90 日涨跌% |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('thscode') or r.get('ticker','')} | "
            f"{r.get('latest_close','-')} | {r.get('latest_date','-')} | "
            f"{r.get('return_30d') if r.get('return_30d') is not None else '-'} | "
            f"{r.get('return_90d') if r.get('return_90d') is not None else '-'} |"
        )
    return "\n".join(lines)


class ComparableAgent(TemplateAgent):
    name = "comparable"
    description = "可比公司估值 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = ComparableScoreCard

    @staticmethod
    def _extract_offering_block(ctx: AgentContext) -> str:
        """从招股书 RAG 抽取「全球发售/价格区间」章节。

        IPO 询价阶段招股价区间通常被 [编纂] 占位, 但其他相关字段（如发行规模、
        基石占比、绿鞋比例）可能已披露, 对估值反推很有价值。
        """
        if ctx.rag is None or not ctx.rag.is_indexed():
            return "（RAG 不可用, 无法抽取招股书「全球发售」章节）"
        queries = [
            "全球发售 价格区间 招股价 招股价区间",
            "发行规模 发行股数 发行总量 绿鞋",
            "基石投资者 基石认购 禁售期",
        ]
        seen = set()
        blocks = []
        for q in queries:
            for h in ctx.rag.search(q, k=4, type_filter="text"):
                key = (h.get("page_start"), h.get("page_end"))
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(
                    f"[P.{h.get('page_start','-')}-{h.get('page_end','-')} "
                    f"| {h.get('section') or ''}]\n{h['text'][:800]}"
                )
        return "\n\n".join(blocks) if blocks else "（RAG 检索未返回相关段落）"

    def build_user_message(self, ctx: AgentContext) -> str:
        peers = ctx.extras.peers
        target = ctx.extras.target_valuation
        target_status_note = (
            ""
            if target
            else "\n⚠️ **target 公司数据缺失**：iFinD 尚未对本项目建档（IPO 询价阶段, "
                 "招股书代码可能尚未分配二级市场流通代码, 或副牌 H 代码未传入）。"
                 "本节估值倍数对比仅含 peers, target 行不出现。"
                 "请基于 peer 倍数 + 招股书披露财务做正向估值, 不要从 iFinD 拉 target 数据。\n"
        )

        # 兜底走旧的 comparable_valuation 工具（基于 list[float] 倍数）
        target_metric = ctx.extras.target_net_profit or ctx.extras.target_revenue or 0
        peer_pe = ctx.extras.peer_pe_multiples or [
            p.get("pe_ttm") for p in peers if p.get("pe_ttm") is not None
        ]
        peer_ps = ctx.extras.peer_ps_multiples or [
            p.get("ps_ttm") for p in peers if p.get("ps_ttm") is not None
        ]
        pe_val = comparable_valuation(target_metric, peer_pe, "PE") if peer_pe else {}
        ps_val = comparable_valuation(target_metric, peer_ps, "PS") if peer_ps else {}

        peers_md = _render_peers_md(peers)
        valuation_md = _render_valuation_md(target, peers)
        fundamentals_md = _render_fundamentals_md(target, peers)
        quotes_md = _render_quotes_md(ctx.extras.peer_recent_quotes)
        offering_block = self._extract_offering_block(ctx)

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n{target_status_note}\n"
            f"# 权威可比公司清单（PeerSuggester + 投决人确认）\n{peers_md}\n\n"
            f"# 招股书「全球发售」章节抽取（用于第四节「招股价区间对比」）\n"
            f"{offering_block}\n\n"
            f"# 估值倍数表（来自 iFinD THS_BD, 截止最新交易日）\n{valuation_md}\n\n"
            f"# 财务基本面表（最近完整年报）\n{fundamentals_md}\n\n"
            f"# 可比公司近期行情（来自 iFinD THS_HQ）\n{quotes_md}\n\n"
            f"# PE 估值结果（基于上方 PE 倍数）\n{pe_val if pe_val else '（无 PE 输入或 target 净利为 0/亏损）'}\n\n"
            f"# PS 估值结果（基于上方 PS 倍数）\n{ps_val if ps_val else '（无 PS 输入或 target 营收为 0）'}\n\n"
            f"请输出可比公司估值分析。"
        )
