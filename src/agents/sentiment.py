from __future__ import annotations

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import SentimentScoreCard
from src.llm import ModelTier

SYSTEM = """你是港股二级市场情绪分析师。基于近期市场表现、新股暗盘情况、同行业公司股价走势，输出：

## 一、所在板块近期情绪（涨跌、成交、北水流向）
## 二、可比已上市公司股价表现（近 30/90 天）
## 三、近期同行业 IPO 暗盘/首日表现
## 四、可比公司近期事件（减持/业绩/回购等会影响赛道情绪的公告）⭐
## 五、本项目路演渠道信号（暗盘价 / 超额认购倍数 / 媒体覆盖密度）⭐
## 六、同期 / 未来 60 天同行业 IPO 队列（资金分流分析）⭐
## 七、市场关注度迹象（媒体覆盖、分析师跟踪）
## 八、对本次基石认购的情绪面判断（顺风/逆风）

输出 1000-1500 字。如缺乏数据，明确指出数据缺口。

【关键约束 — 严禁幻觉】
- 你**没有**联网搜索能力，训练知识截止时间早于本次分析时点。
- 涉及具体股票代码、招股价、上市日期、近期涨跌幅、首日涨跌幅、暗盘价等定量信息，
  必须**严格来自下方"可比公司近期行情"和"近期同行业 IPO 表现"两个数据块**。
- 数据块为空 / 未提供时，必须显式写"该指标当前未提供，下文判断为方向性而非定量"，
  并用方向性词汇（"赛道稀缺""高端制造受关注"）替代具体数字。
- 禁止凭训练记忆编造："越疆 12 月 IPO 首日破发""黑芝麻 8 月 ..."之类的具体年份+涨跌
  描述若不在数据块里就**不得出现**。
- 唯一可凭常识做的判断：板块结构性偏好、北水偏好类型（不带年份与百分比）。
- **是否破发**的判断必须用数据块的"首日开盘价 vs 招股价":
  例 越疆 18.8 → 19.7 = +4.8% 未破发; 优必选 90.0 → 89.9 = -0.1% **首日开盘小幅破发**。
  禁止凭训练记忆说"越疆首日破发"或"优必选大涨"等与数据块矛盾的描述。"""


def _render_peers_quotes(rows: list[dict]) -> str:
    if not rows:
        return "（未提供可比公司行情数据）"
    lines = ["| 代码 | 公司 | 最新价 | 最新日期 | 30 日涨跌% | 90 日涨跌% |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        name = r.get("name") or ""
        lines.append(
            f"| {r.get('thscode') or r.get('ticker','')} | {name} | "
            f"{r.get('latest_close','-')} | {r.get('latest_date','-')} | "
            f"{r.get('return_30d') if r.get('return_30d') is not None else '-'} | "
            f"{r.get('return_90d') if r.get('return_90d') is not None else '-'} |"
        )
    return "\n".join(lines)


def _render_peers_info(rows: list[dict]) -> str:
    """可比公司 IPO 信息: 上市日期 / 招股价 等。"""
    if not rows:
        return "（未提供可比公司 IPO 信息）"
    lines = ["| 代码 | 公司 | 上市日期 | 招股价 | 招股价上限 |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('thscode') or r.get('ticker','')} | {r.get('name','')} | "
            f"{r.get('ipo_date','-')} | {r.get('ipo_price','-')} | {r.get('ipo_max_price','-')} |"
        )
    return "\n".join(lines)


def _render_southbound_md(macro: dict) -> str:
    """渲染近期南向资金趋势(全市场, 来自 EDB)。

    用 SOUTHBOUND_NET_HKD_DAILY (港股通日度净买入 HKD) + SOUTHBOUND_CUM_CNY (累计净买入 CNY) 序列, 取最近 7 个交易日 + 简单统计。
    """
    if not macro:
        return "（未提供宏观 EDB 数据；macro_indicators 为空）"
    daily = macro.get("SOUTHBOUND_NET_HKD_DAILY") or []
    cum = macro.get("SOUTHBOUND_CUM_CNY") or []
    if not daily and not cum:
        return "（macro_indicators 中无南向资金 series）"

    parts: list[str] = []
    # 近 7 期日度净流入
    if daily:
        recent = daily[-7:]
        if recent:
            parts.append("**近 7 个交易日南向资金净流入（万港元）**:\n")
            lines = ["| 日期 | 日度净买入(港币) |", "|---|---|"]
            for r in recent:
                lines.append(f"| {r.get('date','-')} | {r.get('value','-')} |")
            parts.append("\n".join(lines))
            # 7 日累计 + 趋势
            try:
                vals = [float(r.get("value")) for r in recent if r.get("value") is not None]
                if vals:
                    avg7 = sum(vals) / len(vals)
                    last = vals[-1]
                    parts.append(
                        f"\n**7 日均值** {avg7:.0f} 万港元 / **最新一日** {last:.0f} 万港元 "
                        f"({'高于均值' if last > avg7 else '低于均值'})"
                    )
            except (TypeError, ValueError):
                pass
    # 累计净流入
    if cum and len(cum) >= 2:
        latest = cum[-1]
        thirty_back = cum[-min(30, len(cum))]
        try:
            d_30 = float(latest.get("value")) - float(thirty_back.get("value"))
            parts.append(
                f"\n**港股通累计净流入(CNY)** 最新 {latest.get('date')} = {latest.get('value')}; "
                f"近 30 期变动 = {d_30:+.0f}"
            )
        except (TypeError, ValueError):
            pass
    return "\n".join(parts)


def _render_peer_announcements(anns: dict[str, list[dict]]) -> str:
    """渲染 peer 近期公告事件（仅含命中关键标签的，全量太多没意义）。"""
    if not anns:
        return "（未提供 peer 公告数据；REST API 未配置或近期无公告）"
    lines = ["| 代码 | 日期 | 标签 | 标题 |", "|---|---|---|---|"]
    n_total = 0
    for ticker, rows in anns.items():
        for a in rows:
            tags = a.get("tags") or []
            if not tags:
                continue  # 只展示命中关键事件的（减持/业绩/回购/增发/更名/调整）
            n_total += 1
            title = (a.get("title") or "")[:60]
            date = a.get("date", "-")
            lines.append(
                f"| {ticker} | {date} | {','.join(tags)} | {title} |"
            )
    if n_total == 0:
        return "（peer 近 180 天公告中无关键事件: 减持/业绩预警/回购/增发等）"
    return "\n".join(lines)


def _render_roadshow_signals(signals: dict) -> str:
    """渲染用户手输的路演信号（暗盘价 / 超额倍数 / 媒体）。"""
    if not signals:
        return "（未提供路演信号；CLI 可加 --dark-pool-price / --oversubscribe-retail / --press-coverage）"
    lines = []
    if signals.get("dark_pool_price") is not None:
        ipo_low = signals.get("ipo_price_low")
        dp = signals["dark_pool_price"]
        if ipo_low and ipo_low > 0:
            ret = (dp / ipo_low - 1) * 100
            lines.append(f"- **暗盘价**: {dp} HKD (vs 招股价下限 {ipo_low}, {ret:+.1f}%)")
        else:
            lines.append(f"- **暗盘价**: {dp} HKD (招股价未提供, 无法算溢价)")
    if signals.get("oversubscribe_retail_x") is not None:
        lines.append(f"- **散户超额认购**: {signals['oversubscribe_retail_x']}x")
    if signals.get("oversubscribe_intl_x") is not None:
        lines.append(f"- **国际配售超额**: {signals['oversubscribe_intl_x']}x")
    if signals.get("press_coverage_score") is not None:
        lines.append(f"- **媒体覆盖热度** (1-5): {signals['press_coverage_score']}")
    if signals.get("press_coverage_notes"):
        lines.append(f"- **媒体覆盖说明**: {signals['press_coverage_notes']}")
    return "\n".join(lines) if lines else "（路演信号字典为空）"


def _render_competing_ipos(rows: list[dict]) -> str:
    """渲染同期 / 未来 60 天同行业其它 IPO."""
    if not rows:
        return "（未提供同期 / 未来 60 天同行业 IPO 数据）"
    lines = ["| 代码 | 公司 | 拟上市日 | 估值规模 | 资金分流影响 |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('thscode') or r.get('ticker','')} | {r.get('name','-')} | "
            f"{r.get('expected_listing_date','-')} | {r.get('expected_marketcap_hkd_b','-')} 亿 | "
            f"{r.get('overlap_note','-')} |"
        )
    return "\n".join(lines)


def _render_recent_ipos(rows: list[dict]) -> str:
    """近期同行业 IPO 表: 招股价 / 首日开盘价 / 首日开盘涨跌。"""
    if not rows:
        return "（未提供近期同行业 IPO 数据）"
    lines = ["| 代码 | 公司 | 上市日期 | 招股价 | 招股价上限 | 首日开盘价 | 首日开盘涨跌% |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        fdo = r.get("first_day_open")
        fdor = r.get("first_day_open_return")
        lines.append(
            f"| {r.get('thscode') or r.get('ticker','')} | {r.get('name','')} | "
            f"{r.get('ipo_date','-')} | {r.get('ipo_price','-')} | "
            f"{r.get('ipo_max_price','-')} | "
            f"{fdo if fdo is not None else '-'} | "
            f"{('+' + str(fdor)) if (fdor is not None and fdor >= 0) else (str(fdor) if fdor is not None else '-')} |"
        )
    return "\n".join(lines)


class SentimentAgent(TemplateAgent):
    name = "sentiment"
    description = "二级市场情绪 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = SentimentScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        peers_quotes_md = _render_peers_quotes(ctx.extras.peer_recent_quotes)
        peers_info_md = _render_peers_info(ctx.extras.peers)
        recent_ipos_md = _render_recent_ipos(ctx.extras.recent_hk_ipos)
        southbound_md = _render_southbound_md(ctx.extras.macro_indicators)
        peer_anns_md = _render_peer_announcements(ctx.extras.peer_announcements)
        roadshow_md = _render_roadshow_signals(ctx.extras.roadshow_signals)
        competing_md = _render_competing_ipos(ctx.extras.competing_ipos)

        # 稀缺性 brief (来自 ScarcityAgent, 上游运行已写入 ctx.briefs)
        scarcity_brief = ctx.briefs.get("scarcity") or "（未运行 ScarcityAgent 或无产出）"
        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 上游稀缺性判断 (来自 ScarcityAgent — 用于'稀缺×情绪'联动判断)\n"
            f"{scarcity_brief}\n\n"
            f"# 可比公司 IPO 信息（来自 iFinD THS_BD）\n{peers_info_md}\n\n"
            f"# 可比公司近期行情（来自 iFinD THS_HQ，最新交易日数据）\n{peers_quotes_md}\n\n"
            f"# 可比公司近 180 天关键公告事件（来自 iFinD report_query）\n{peer_anns_md}\n\n"
            f"# 近期同行业 IPO 表现\n{recent_ipos_md}\n\n"
            f"# 本项目路演信号（用户手输 / iFinD 拉不到）\n{roadshow_md}\n\n"
            f"# 同期 / 未来 60 天同行业 IPO 队列（资金分流）\n{competing_md}\n\n"
            f"# 港股通南向资金（来自 iFinD EDB, 全市场口径）\n{southbound_md}\n\n"
            f"请输出情绪分析。第一节'板块近期情绪'必须引用上方南向资金数据"
            f"（如近 7 日净流入趋势 / 7 日均值 vs 最新一日对比 / 累计净流入变动）, 不要写'未提供'。"
            f"\n\n**稀缺×情绪联动**: 如 ScarcityAgent 给出'稀缺+热情', 你需要在第八节"
            f"'对基石认购的情绪面判断'里说明溢价空间; 若'拥挤+冷淡', 必须明确转向负面。"
        )
