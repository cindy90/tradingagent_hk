from __future__ import annotations

from typing import Any

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.feedback.models import MacroScoreCard
from src.llm import ModelTier


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _render_indices_md(rows: list[dict]) -> str:
    """渲染港股指数表 (来自 iFinD)。"""
    if not rows:
        return "（iFinD 未返回指数数据；请检查 IFIND_USERNAME/PASSWORD 配置）"
    lines = ["| 指数 | 最新点位 | 最新日期 | 30 日变动% | 90 日变动% | PE-TTM | PB |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('index')} ({r.get('thscode','-')}) | "
            f"{r.get('latest_close','-')} | {r.get('latest_date','-')} | "
            f"{r.get('change_30d') if r.get('change_30d') is not None else '-'} | "
            f"{r.get('change_90d') if r.get('change_90d') is not None else '-'} | "
            f"{r.get('pe_ttm') if r.get('pe_ttm') is not None else '-'} | "
            f"{r.get('pb_latest') if r.get('pb_latest') is not None else '-'} |"
        )
    return "\n".join(lines)


def _percentile_rank(series: list[dict], latest_value: float | None) -> tuple[float | None, float | None, float | None]:
    """计算 latest_value 在 series 中的百分位 (0-100) + min/max。

    Returns: (percentile, min, max)
    分位定义：series 中严格小于 latest 的样本占比。
    """
    if latest_value is None:
        return (None, None, None)
    vals: list[float] = []
    for p in series:
        v = _to_float(p.get("value"))
        if v is not None:
            vals.append(v)
    if len(vals) < 10:  # 样本太少不做分位
        return (None, None, None)
    below = sum(1 for v in vals if v < latest_value)
    return (round(below / len(vals) * 100, 1), round(min(vals), 4), round(max(vals), 4))

SYSTEM = """你是港股市场宏观策略分析师。基于当前宏观环境，分析对本 IPO 基石认购的影响：

## 一、全球与中国宏观（美联储路径、人民币汇率、中国经济周期）
## 二、香港市场流动性（HIBOR、IPO 集资额、北水南下情况、新股认购热度）
## 3、恒生指数估值与情绪状态（PE 分位、波动率）
## 四、港股 IPO 市场近期表现（破发率、首日涨跌、暗盘表现）
## 五、对本项目所属板块的偏好倾向
## 六、宏观环境对基石策略的总体判断（窗口好/坏，1-5 分）

要求：使用提供的市场数据，给出有数字支撑的判断。输出 1200-1800 字。

【关键约束 — 严禁幻觉】
- 你**没有**联网搜索能力，你的训练知识截止时间早于本次分析的项目时点。
- 任何带具体年份的"宏观历史数据"（例如"2024 年 HIBOR 5.5%""2024 年港股 IPO 集资 800 亿""2024 年恒指 PE 9.5x"）
  必须**严格来自下方"宏观指标"块的真实数据**。该块若为空或某指标缺失，**禁止编造**。
- 缺数据时，请明确写"该指标当前未提供，下文判断为方向性而非定量"，并用方向性词汇（"流动性边际改善""估值偏低分位"）替代具体数字。
- 唯一可凭常识推断的内容：项目所属板块的**结构性偏好**（如"机器人赛道近年受关注"），但不要带年份和百分比。
- 引用恒指点位时只用下方"当前恒生指数"块中的实时值，不要写"基于历史推断"的具体数字。"""


class MacroAgent(TemplateAgent):
    name = "macro"
    description = "宏观策略 Agent"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = MacroScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        # 港股三大指数: 优先从 ctx.extras.market_indices (workflow 已 prefetch),
        # 缺失时实时调一次 iFinD（兼容 rerun 场景）
        indices = ctx.extras.market_indices
        if not indices:
            try:
                from src.data.ifind_sdk import get_indices_summary
                indices = get_indices_summary(["HSI", "HSCEI", "HSTECH"])
            except Exception as e:
                from loguru import logger as _lg
                _lg.warning(f"get_indices_summary 失败: {e}")
                indices = []
        macro = ctx.extras.macro_indicators

        # 把 EDB 时间序列压缩为"最新值 / 6 期前 / 12 期前 / 历史分位 / 趋势"
        # 历史分位仅对部分有意义的指标（PE/HIBOR/汇率等连续型指标）展示, 月频指标
        # 不展示 (CN_PMI 月频, 12 期 = 12 个月; 日频如 HIBOR 12 期 = 12 个工作日)
        PERCENTILE_INDICATORS = {
            "HSI_PE_TTM",      # 恒指 PE 分位（核心估值锚）
            "HIBOR_1M", "HIBOR_3M",  # 利率历史分位
            "USD_HKD_FWD_ON",   # 港元远期点位分位
        }
        macro_lines = []
        for name, series in macro.items():
            if not series:
                macro_lines.append(f"- {name}: (无数据)")
                continue
            latest = series[-1]
            prev6 = series[-min(6, len(series))] if len(series) > 1 else None
            prev12 = series[-min(12, len(series))] if len(series) > 1 else None
            parts = [f"最新({latest.get('date')}): {latest.get('value')}"]
            if prev6:
                parts.append(f"6 期前: {prev6.get('value')}")
            if prev12 and prev12 is not prev6:
                parts.append(f"12 期前: {prev12.get('value')}")
            # 加历史分位
            if name in PERCENTILE_INDICATORS:
                latest_v = _to_float(latest.get("value"))
                pct, lo, hi = _percentile_rank(series, latest_v)
                if pct is not None:
                    parts.append(
                        f"**历史分位 {pct}%** (序列 N={len(series)}, "
                        f"区间 {lo}—{hi})"
                    )
            macro_lines.append(f"- {name}: " + " | ".join(parts))
        macro_block = "\n".join(macro_lines) if macro_lines else "(同花顺 EDB 未返回数据)"

        # 港股 IPO 破发率（基于 ctx.extras.recent_hk_ipos 已有的 first_day_open 数据）
        ipo_break_block = self._compute_ipo_break_rate_block(ctx)

        return (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  行业: {ctx.industry}\n\n"
            f"# 港股三大指数（来自 iFinD THS_HQ + THS_BD，最新交易日数据）\n"
            f"{_render_indices_md(indices)}\n\n"
            f"# 宏观指标（同花顺 EDB, 含历史分位）\n{macro_block}\n\n"
            f"# 近期港股 IPO 首日破发统计\n{ipo_break_block}\n\n"
            f"请输出宏观策略分析。第三节'恒指估值'必须引用上方 HSI PE 数据 + 历史分位（若 EDB 有），"
            f"第五节'板块偏好'引用三大指数 30/90 日变动对比。"
        )

    @staticmethod
    def _compute_ipo_break_rate_block(ctx: AgentContext) -> str:
        """从 ctx.extras.recent_hk_ipos 计算近期 IPO 首日开盘破发率."""
        rows = ctx.extras.recent_hk_ipos
        if not rows:
            return "（recent_hk_ipos 为空; 可在 cli analyze 加 --recent-ipos 参数提供 IPO 队列）"
        total = 0
        broken = 0
        below_max = 0
        details = []
        for r in rows:
            ipo = _to_float(r.get("ipo_price"))
            fdo = _to_float(r.get("first_day_open"))
            if ipo is None or fdo is None or ipo <= 0:
                continue
            total += 1
            ret = (fdo / ipo - 1) * 100
            if fdo < ipo:
                broken += 1
            if r.get("ipo_max_price") and _to_float(r.get("ipo_max_price")) is not None:
                if ipo < _to_float(r.get("ipo_max_price")):
                    below_max += 1
            details.append(
                f"  - {r.get('thscode') or r.get('ticker','')} "
                f"{r.get('name','')}: 招股 {ipo} → 首日开 {fdo} ({ret:+.2f}%) "
                f"{'破发' if fdo < ipo else '高开'}"
            )
        if total == 0:
            return "（recent_hk_ipos 中无含完整 ipo_price+first_day_open 的样本）"
        rate = round(broken / total * 100, 1)
        head = (
            f"**样本数 {total} 家, 首日开盘破发 {broken} 家 → "
            f"破发率 {rate}%**; 折价定价(招股价低于上限) {below_max}/{total} 家。"
        )
        return head + "\n" + "\n".join(details)
