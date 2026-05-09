"""ValuationEngine — 确定性的多方法估值计算引擎。

设计动机:
之前估值是 LLM 算 (PS × 营收 / PE × 净利 等), 经常算错小数点 / 单位 / 加权。
本引擎把"算数"从 LLM 移到 Python, LLM 只提供:
- 输入参数: 用哪个 PE / 用哪个 CAGR / 用哪个 PS
- rationale: 为什么这么选
- 业务相似度 / 商业模式判断

引擎做:
- 应用各估值方法的标准公式
- 按 ListingProfile 强制方法选择 (18A 必须 rNPV, AH 必须折价锚)
- 自动加权综合
- 单位规范化 (RMB → HKD 汇率)
- IPO 折扣应用
- Sanity check (估值 < 0 / 倍数异常)

LLM 输入: 方法选择 + 倍数 + 财务输入 + rationale
LLM 不做: 加权求和 / 单位换算 / 区间生成

支持的估值方法:
- PE   : PE × 净利润 (盈利公司)
- PS   : PS × 营收 (亏损/早期成长)
- PEG  : (PE / CAGR) × CAGR × 净利润 ← 简化为 PEG_target × CAGR × 净利
- PB   : PB × 净资产
- EV/EBITDA, EV/Sales : 同理
- DCF  : 简化 5 年贴现 (LLM 提供 FCF 假设)
- rNPV : 18A 风险调整 NPV (Peak Sales × 上市概率)
- SOTP : 多业务线分别估值合计
- AH_Anchor : A 股市值 × 港股折价
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ============================ 输入数据结构 ============================

@dataclass
class ValuationMethodInput:
    """LLM 提供的单一方法输入参数. 引擎据此计算 result_hkd_b."""
    method: str   # "PE" / "PS" / "PEG" / "PB" / "EV/EBITDA" / "EV/Sales" /
                  # "DCF" / "rNPV" / "SOTP" / "AH_Anchor"
    # 输入参数 (可选, 视方法而定)
    multiple: float | None = None     # PE / PS / PB / EV-EBITDA / EV-Sales / PEG_target
    target_metric: float | None = None  # 净利 / 营收 / 净资产 / EBITDA, 默认 RMB 单位
    # PEG 专用
    cagr: float | None = None  # 增长率, 0.35 表示 35%
    # DCF 专用
    fcf_5y: list[float] | None = None  # 5 年 FCF 预测, 单位与 target_metric 一致
    discount_rate: float | None = None  # WACC, 0.10 = 10%
    terminal_growth: float | None = None  # 永续增长, 0.03 = 3%
    # rNPV 专用 (18A)
    peak_sales: float | None = None  # 销售峰值
    peak_sales_multiple: float | None = None  # 销售峰值 → 市值倍数 (业内通常 3-6x)
    success_probability: float | None = None  # 上市概率, 0.6 = 60%
    # SOTP 专用 (子业务列表)
    sotp_components: list[dict] | None = None
        # [{"name": "硬件", "value_hkd_b": 40, "weight": 0.5}, ...]
    # AH 专用
    a_share_market_cap_hkd_b: float | None = None  # A 股市值（已换算 HKD）
    ah_discount: float | None = None  # 折价比例, 0.30 = 港股低 30%
    # 通用
    rmb_to_hkd: float = 1.10  # RMB → HKD 汇率, 默认 1.1
    weight: float = 1.0  # 该方法在综合估值中的权重 (0-1)
    rationale: str = ""
    peer_basis: str = ""


# ============================ 输出 ============================

@dataclass
class ValuationMethodResult:
    method: str
    result_hkd_b: float | None
    formula: str    # 自动生成的公式描述
    weight: float
    peer_basis: str
    target_metric: str  # 自动生成的输入参数描述
    rationale: str
    sanity_warnings: list[str] = field(default_factory=list)


@dataclass
class ValuationRange:
    """三档估值 + 综合."""
    low_hkd_b: float | None
    mid_hkd_b: float
    high_hkd_b: float | None
    anchor_method: str   # 加权综合 / 单方法
    weighted_methods: list[ValuationMethodResult]
    sum_of_weights: float
    overall_warnings: list[str] = field(default_factory=list)


# ============================ 各方法的实现 ============================

def _to_hkd_b(value: float | None, rmb_to_hkd: float = 1.10) -> float | None:
    """单位换算: 输入数字 (RMB 亿) × 汇率 → HKD 亿. 输入已是 HKD 时 caller 自己设 rmb_to_hkd=1.0."""
    if value is None or value <= 0:
        return None
    return round(value * rmb_to_hkd, 4)


def _apply_pe(inp: ValuationMethodInput) -> ValuationMethodResult:
    warnings = []
    multiple = inp.multiple
    metric = inp.target_metric  # 净利润, 默认 RMB
    if multiple is None or metric is None or metric <= 0:
        return ValuationMethodResult(
            method="PE", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="净利润缺失", rationale=inp.rationale,
            sanity_warnings=["PE 法需要 multiple + target_metric (净利润 > 0)"],
        )
    if multiple > 200:
        warnings.append(f"PE={multiple} 异常高 (> 200), 建议复核可比")
    if multiple < 5:
        warnings.append(f"PE={multiple} 异常低 (< 5), 建议复核可比")

    raw = multiple * metric
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = f"PE {multiple} × 净利润 {metric} 亿 RMB = {raw:.2f} 亿 RMB ≈ {result} 亿 HKD"
    return ValuationMethodResult(
        method="PE", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis, target_metric=f"净利润 {metric} 亿 RMB",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_ps(inp: ValuationMethodInput) -> ValuationMethodResult:
    warnings = []
    multiple = inp.multiple
    metric = inp.target_metric  # 营收
    if multiple is None or metric is None or metric <= 0:
        return ValuationMethodResult(
            method="PS", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="营收缺失", rationale=inp.rationale,
            sanity_warnings=["PS 法需要 multiple + target_metric (营收 > 0)"],
        )
    if multiple > 50:
        warnings.append(f"PS={multiple} 异常高 (> 50), 建议复核")

    raw = multiple * metric
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = f"PS {multiple} × 营收 {metric} 亿 RMB = {raw:.2f} 亿 RMB ≈ {result} 亿 HKD"
    return ValuationMethodResult(
        method="PS", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis, target_metric=f"营收 {metric} 亿 RMB",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_peg(inp: ValuationMethodInput) -> ValuationMethodResult:
    """PEG 法: 给定 PEG_target (1.0-2.0 合理), 算 implied PE = PEG × CAGR%, 再 × 净利."""
    warnings = []
    peg = inp.multiple  # PEG 目标值, 1.0-2.0 通常
    cagr = inp.cagr     # 0.35
    metric = inp.target_metric  # 净利

    if peg is None or cagr is None or metric is None or metric <= 0:
        return ValuationMethodResult(
            method="PEG", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="缺 PEG_target / CAGR / 净利",
            rationale=inp.rationale,
            sanity_warnings=["PEG 法需要 multiple (PEG目标值) + cagr + target_metric (净利)"],
        )
    if cagr <= 0:
        warnings.append(f"CAGR={cagr*100:.0f}% ≤ 0, PEG 法不适用 (增长公司专用)")
    if peg > 3:
        warnings.append(f"PEG={peg} > 3, 估值偏高警示")

    implied_pe = peg * cagr * 100  # cagr=0.35, peg=1.4 → implied PE = 1.4 × 35 = 49
    raw = implied_pe * metric
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = (
        f"PEG {peg} × CAGR {cagr*100:.0f}% = implied PE {implied_pe:.1f}; "
        f"× 净利 {metric} 亿 RMB = {raw:.2f} 亿 RMB ≈ {result} 亿 HKD"
    )
    return ValuationMethodResult(
        method="PEG", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis,
        target_metric=f"PEG {peg} × CAGR {cagr*100:.0f}% × 净利 {metric} 亿",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_simple_multiple(inp: ValuationMethodInput, name: str, metric_label: str) -> ValuationMethodResult:
    """通用倍数法 (PB / EV-EBITDA / EV-Sales)."""
    warnings = []
    multiple = inp.multiple
    metric = inp.target_metric
    if multiple is None or metric is None or metric <= 0:
        return ValuationMethodResult(
            method=name, result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric=f"{metric_label}缺失", rationale=inp.rationale,
            sanity_warnings=[f"{name} 法需要 multiple + target_metric ({metric_label} > 0)"],
        )
    raw = multiple * metric
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = f"{name} {multiple} × {metric_label} {metric} = {raw:.2f} 亿 ≈ {result} 亿 HKD"
    return ValuationMethodResult(
        method=name, result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis, target_metric=f"{metric_label} {metric} 亿",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_dcf(inp: ValuationMethodInput) -> ValuationMethodResult:
    """简化 DCF: 5 年 FCF 贴现 + 永续增长终值."""
    warnings = []
    fcf = inp.fcf_5y
    r = inp.discount_rate
    g = inp.terminal_growth or 0.03
    if not fcf or len(fcf) < 3 or r is None:
        return ValuationMethodResult(
            method="DCF", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="DCF 输入不全", rationale=inp.rationale,
            sanity_warnings=["DCF 法需要 fcf_5y (>=3 年) + discount_rate"],
        )
    if r <= g:
        warnings.append(f"折现率 {r} ≤ 永续增长 {g}, DCF 公式失效")
        return ValuationMethodResult(
            method="DCF", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="折现率 ≤ 永续增长", rationale=inp.rationale,
            sanity_warnings=warnings,
        )

    # 5 年 FCF 贴现
    pv_fcf = sum(f / ((1 + r) ** (i + 1)) for i, f in enumerate(fcf))
    # 终值: TV = FCF_n × (1+g) / (r-g), 贴现到第 n 年末
    n = len(fcf)
    tv = fcf[-1] * (1 + g) / (r - g)
    pv_tv = tv / ((1 + r) ** n)
    raw = pv_fcf + pv_tv
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = (
        f"5年FCF贴现 {pv_fcf:.2f} + 终值贴现 {pv_tv:.2f} = {raw:.2f} 亿 RMB ≈ {result} 亿 HKD "
        f"(折现率 {r*100:.0f}%, 永续 {g*100:.0f}%)"
    )
    return ValuationMethodResult(
        method="DCF", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis,
        target_metric=f"FCF[{','.join(f'{x:.1f}' for x in fcf)}], r={r}, g={g}",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_rnpv(inp: ValuationMethodInput) -> ValuationMethodResult:
    """rNPV (18A): Peak Sales × 销售倍数 × 上市概率."""
    warnings = []
    peak = inp.peak_sales
    mult = inp.peak_sales_multiple
    prob = inp.success_probability
    if peak is None or mult is None or prob is None:
        return ValuationMethodResult(
            method="rNPV", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="rNPV 输入不全", rationale=inp.rationale,
            sanity_warnings=["rNPV 法需要 peak_sales + peak_sales_multiple + success_probability"],
        )
    if not 0 < prob <= 1:
        warnings.append(f"success_probability={prob} 不在 (0, 1] 区间")
    if mult > 10:
        warnings.append(f"peak_sales_multiple={mult} > 10, 异常高")

    raw = peak * mult * prob
    result = _to_hkd_b(raw, inp.rmb_to_hkd)
    formula = (
        f"Peak Sales {peak} 亿 × 销售倍数 {mult} × 上市概率 {prob*100:.0f}% "
        f"= {raw:.2f} 亿 RMB ≈ {result} 亿 HKD"
    )
    return ValuationMethodResult(
        method="rNPV", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis,
        target_metric=f"Peak Sales {peak}, 倍数 {mult}, 概率 {prob}",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_sotp(inp: ValuationMethodInput) -> ValuationMethodResult:
    """SOTP: 多业务线估值合计."""
    warnings = []
    comps = inp.sotp_components
    if not comps:
        return ValuationMethodResult(
            method="SOTP", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="SOTP 子业务为空", rationale=inp.rationale,
            sanity_warnings=["SOTP 法需要 sotp_components 列表"],
        )
    sub_results: list[str] = []
    total = 0.0
    for c in comps:
        name = c.get("name", "")
        value = c.get("value_hkd_b") or 0
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        total += v
        sub_results.append(f"{name} {v}")

    if total <= 0:
        return ValuationMethodResult(
            method="SOTP", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="SOTP 合计 ≤ 0", rationale=inp.rationale,
            sanity_warnings=["SOTP 子业务合计 ≤ 0"],
        )

    formula = " + ".join(sub_results) + f" = {total:.2f} 亿 HKD"
    return ValuationMethodResult(
        method="SOTP", result_hkd_b=round(total, 4), formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis, target_metric=f"{len(comps)} 个子业务",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


def _apply_ah_anchor(inp: ValuationMethodInput) -> ValuationMethodResult:
    """AH 双重: A 股市值 × (1 - 港股折价)."""
    warnings = []
    a_cap = inp.a_share_market_cap_hkd_b
    discount = inp.ah_discount
    if a_cap is None or discount is None:
        return ValuationMethodResult(
            method="AH_Anchor", result_hkd_b=None, formula="—", weight=inp.weight,
            peer_basis=inp.peer_basis, target_metric="AH 锚定输入不全", rationale=inp.rationale,
            sanity_warnings=["AH_Anchor 需要 a_share_market_cap_hkd_b + ah_discount"],
        )
    if not 0 <= discount <= 0.6:
        warnings.append(f"ah_discount={discount} 不在 [0, 0.6] 历史经验区间")

    result = round(a_cap * (1 - discount), 4)
    formula = (
        f"A 股市值 {a_cap} 亿 HKD × (1 - 港股折价 {discount*100:.0f}%) = {result} 亿 HKD"
    )
    return ValuationMethodResult(
        method="AH_Anchor", result_hkd_b=result, formula=formula, weight=inp.weight,
        peer_basis=inp.peer_basis,
        target_metric=f"A 股 {a_cap} 亿 × 折价 {discount*100:.0f}%",
        rationale=inp.rationale, sanity_warnings=warnings,
    )


_METHOD_DISPATCH = {
    "PE": _apply_pe,
    "PS": _apply_ps,
    "PEG": _apply_peg,
    "PB": lambda i: _apply_simple_multiple(i, "PB", "净资产"),
    "EV/EBITDA": lambda i: _apply_simple_multiple(i, "EV/EBITDA", "EBITDA"),
    "EV/Sales": lambda i: _apply_simple_multiple(i, "EV/Sales", "营收"),
    "DCF": _apply_dcf,
    "rNPV": _apply_rnpv,
    "SOTP": _apply_sotp,
    "AH_Anchor": _apply_ah_anchor,
}


def apply_method(inp: ValuationMethodInput) -> ValuationMethodResult:
    """单一方法应用. 未知方法返回 None 结果 + warning."""
    fn = _METHOD_DISPATCH.get(inp.method)
    if fn is None:
        return ValuationMethodResult(
            method=inp.method, result_hkd_b=None, formula="—",
            weight=inp.weight, peer_basis=inp.peer_basis,
            target_metric="未知方法", rationale=inp.rationale,
            sanity_warnings=[f"未知估值方法 {inp.method}, 支持: {list(_METHOD_DISPATCH)}"],
        )
    return fn(inp)


# ============================ 综合 ============================

def compute_valuation_range(
    inputs: list[ValuationMethodInput],
    *,
    forbidden_methods: set[str] | None = None,
    ipo_discount_pct: float | None = None,
    low_high_factor: tuple[float, float] = (0.85, 1.15),
) -> ValuationRange:
    """对多个方法求加权综合 + 三档估值区间.

    Args:
        inputs: LLM 提供的方法输入列表
        forbidden_methods: ListingProfile 决定的禁用方法 (例: 18A 禁 PE)
        ipo_discount_pct: 港股 IPO 折扣 (0.15 = 15%)
        low_high_factor: 区间因子, 默认 mid × 0.85 / × 1.15
    """
    overall_warnings: list[str] = []
    forbidden = forbidden_methods or set()

    results: list[ValuationMethodResult] = []
    for inp in inputs:
        if inp.method in forbidden:
            overall_warnings.append(
                f"忽略禁用方法 {inp.method} (ListingProfile 不支持)"
            )
            continue
        results.append(apply_method(inp))

    valid = [r for r in results if r.result_hkd_b is not None and r.result_hkd_b > 0]
    if not valid:
        return ValuationRange(
            low_hkd_b=None, mid_hkd_b=0.0, high_hkd_b=None,
            anchor_method="无有效方法",
            weighted_methods=results, sum_of_weights=0.0,
            overall_warnings=overall_warnings + ["所有方法计算失败或被禁用"],
        )

    sum_w = sum(r.weight for r in valid)
    if sum_w <= 0:
        # 等权回退
        sum_w = float(len(valid))
        for r in valid:
            r.weight = 1.0
    if not (0.92 <= sum_w <= 1.08):
        overall_warnings.append(
            f"方法权重之和 {sum_w:.2f} 偏离 1.0, 已自动归一化"
        )

    weighted_sum = sum(r.result_hkd_b * r.weight for r in valid) / sum_w
    mid = round(weighted_sum, 4)

    if ipo_discount_pct is not None and 0 < ipo_discount_pct < 0.5:
        # IPO 折扣应用到中枢
        mid_pre = mid
        mid = round(mid * (1 - ipo_discount_pct), 4)
        overall_warnings.append(
            f"应用 IPO 折扣 {ipo_discount_pct*100:.0f}%: {mid_pre} → {mid} 亿 HKD"
        )

    low = round(mid * low_high_factor[0], 4)
    high = round(mid * low_high_factor[1], 4)

    anchor = "加权综合" if len(valid) > 1 else valid[0].method

    # 警告聚合
    for r in valid:
        for w in r.sanity_warnings:
            overall_warnings.append(f"[{r.method}] {w}")

    return ValuationRange(
        low_hkd_b=low, mid_hkd_b=mid, high_hkd_b=high,
        anchor_method=anchor, weighted_methods=results,
        sum_of_weights=round(sum_w, 4),
        overall_warnings=overall_warnings,
    )


def select_methods_by_profile(profile: Any) -> dict[str, set[str]]:
    """根据 ListingProfile 返回 {primary, forbidden} 方法集合 (字符串)。

    委托给 listing_profile.recommended_valuation_methods, 但提取为 set 便于约束。
    """
    if profile is None:
        return {"primary": set(), "forbidden": set()}
    try:
        from src.agents.listing_profile import recommended_valuation_methods
    except ImportError:
        return {"primary": set(), "forbidden": set()}

    methods = recommended_valuation_methods(profile)
    # 把"rNPV (管线风险调整 NPV)" 等描述里抽出关键词
    primary_raw = methods.get("primary", []) + methods.get("secondary", [])
    forbidden_raw = methods.get("forbidden", [])

    def _extract_methods(raw_list: list[str]) -> set[str]:
        # 用 word boundary 严格匹配, 避免 "PE" in "Peak Sales" 误判
        import re
        keys: set[str] = set()
        # 关键词 + 对应输出名 (LLM 写 "rNPV" 我们要存 "rNPV", 不是 "RNPV")
        kw_map = {
            r"\bPEG\b": "PEG",
            r"\bPE-TTM\b": "PE",
            r"\bPE\b(?!G)(?!AK)": "PE",  # PE 但不是 PEG / PEAK 开头
            r"\bPS\b": "PS",
            r"\bPB\b": "PB",
            r"\bDCF\b": "DCF",
            r"\brNPV\b": "rNPV",
            r"\bRNPV\b": "rNPV",
            r"\bSOTP\b": "SOTP",
            r"EV[/-]EBITDA": "EV/EBITDA",
            r"EV[/-]Sales": "EV/Sales",
            r"\bA-H\b": "AH_Anchor",
            r"AH[ _]Anchor": "AH_Anchor",
        }
        for s in raw_list:
            for pattern, out_key in kw_map.items():
                if re.search(pattern, s, re.IGNORECASE):
                    keys.add(out_key)
        return keys

    return {
        "primary": _extract_methods(primary_raw),
        "forbidden": _extract_methods(forbidden_raw),
    }
