"""上市档案（ListingProfile）+ profile-aware 调权 / 估值方法 / 风险维度规则。

设计动机:
当前 STANDARD_DECISION_FACTORS 是通用 6 因子模板, 不区分:
- 18A 未盈利生物科技 (估值方法应用 rNPV / Peak Sales × 概率)
- 18C 特专科技 (估值方法应用 PS / EV-Sales / DCF, PE 不适用)
- AH 双重 / 二次上市 (估值锚定 A 股或主上市市场 × 折价)
- WVR 同股不同权 (治理风险加权)
- 不同规模档 (小盘 → 流动性权重升级)

本模块把这些差异化逻辑显式化:
1. ListingProfile 描述项目特征 (上市规则章节 / 盈利状态 / 规模 / 行业 / 治理)
2. adjusted_weight_ranges(profile) 给 STANDARD_DECISION_FACTORS 调权
3. recommended_valuation_methods(profile) 强约束估值方法 (硬性)
4. extra_risk_dimensions(profile) 风险维度增补 (例如 18A 加临床失败维度)
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------- ListingProfile ----------------------------

class ListingProfile(BaseModel):
    """港股 IPO 项目的上市档案. 决定下游差异化处理路径。

    本模型字段尽量贴合 HKEX 上市规则的真实分类, 不做过度抽象。
    """
    model_config = ConfigDict(extra="ignore")

    listing_chapter: Literal[
        "Main_Board_Standard",   # 主板标准（盈利测试）
        "Main_Board_18A",        # 未盈利生物科技
        "Main_Board_18C",        # 特专科技
        "Main_Board_19C",        # 不同投票权（WVR）
        "Secondary_Listing",     # 二次上市（中概股回归等, 主上市仍在美股）
        "Dual_Primary_AH",       # AH 双重主要上市
        "GEM",                   # 创业板（已边缘化, 历史项目）
        "Unknown",
    ] = Field(default="Unknown", description="HKEX 上市规则适用章节")

    profitability_stage: Literal[
        "Profitable_Stable",     # 已稳定盈利, PE 适用
        "Profitable_Growth",     # 盈利但高速增长, PE/PEG 适用
        "Pre_Profit_Late_Stage", # 临近盈利, PS/EV-Sales 主导
        "Loss_Making_Growth",    # 亏损但高速成长, PS 主导
        "Pre_Commercial",        # 18A/18C 未商业化, DCF/管线估值
        "Unknown",
    ] = Field(default="Unknown")

    size_tier: Literal[
        "Small",   # < 30 亿 HKD
        "Mid",     # 30-300 亿
        "Large",   # 300-1000 亿
        "Mega",    # > 1000 亿
        "Unknown",
    ] = Field(default="Unknown")

    industry_theme: Literal[
        "Tech_AI_Semi",          # AI / 半导体 / 通用科技
        "Bio_Pharma",            # 生物医药 (18A 主战场)
        "Med_Device",            # 医疗器械
        "Robotics_Automation",   # 机器人 / 工业自动化 (18C 候选)
        "New_Energy",            # 新能源 (18C 候选)
        "Advanced_Materials",    # 先进材料 (18C 候选)
        "Consumer",              # 消费
        "Financial",             # 金融
        "Real_Estate",           # 地产
        "Industrial",            # 工业 / 制造
        "Healthcare",            # 医疗服务（非药）
        "Other",
    ] = Field(default="Other")

    # 治理特征
    has_wvr: bool = Field(default=False, description="是否同股不同权")
    has_a_share_listed: bool = Field(default=False, description="是否已在 A 股上市")
    a_share_ticker: str = Field(default="", description="A 股代码（如有）")
    is_concept_stock: bool = Field(default=False, description="是否中概股")
    main_listing_market: str = Field(default="",
        description="二次上市时主上市地, 例: 'NASDAQ' / 'NYSE'")

    detection_confidence: Literal["显式确认", "招股书推断", "默认值"] = "默认值"
    detection_evidence: list[str] = Field(
        default_factory=list,
        description="推断证据（招股书页码 / 关键句子）",
    )

    @property
    def is_unprofitable_listing(self) -> bool:
        """是否走 18A / 18C 等未盈利通道."""
        return self.listing_chapter in ("Main_Board_18A", "Main_Board_18C") or \
               self.profitability_stage == "Pre_Commercial"

    @property
    def needs_a_share_anchor(self) -> bool:
        """是否需要 A 股锚定估值."""
        return self.has_a_share_listed or \
               self.listing_chapter == "Dual_Primary_AH"

    @property
    def is_secondary_listing(self) -> bool:
        return self.listing_chapter == "Secondary_Listing"


# ---------------------------- 调权矩阵 ----------------------------

# (factor_name, adjustment_to_lower, adjustment_to_upper, rationale)
# 增量调整: 加到 STANDARD_DECISION_FACTORS 的 suggested_weight_range 上
# 例: ("业务质量", +0.10, +0.10) = lower/upper 各加 0.10

ProfileAdjustment = tuple[str, float, float, str]


def adjusted_weight_ranges(profile: ListingProfile) -> list[dict[str, Any]]:
    """根据 listing_profile 调整 STANDARD_DECISION_FACTORS 的权重区间.

    返回与 STANDARD_DECISION_FACTORS 同构的列表, 但 suggested_weight_range
    根据 profile 调整后, 并附 profile_adjustments 字段记录调整原因。
    """
    # 延迟 import 避免循环 (decision.py imports listing_profile.py 通过 prompt 渲染)
    from src.agents.decision import STANDARD_DECISION_FACTORS

    base = [dict(f) for f in STANDARD_DECISION_FACTORS]
    adjustments: list[ProfileAdjustment] = []

    # ---- 18A 未盈利生物科技 ----
    if profile.listing_chapter == "Main_Board_18A":
        adjustments += [
            ("业务质量", +0.10, +0.10,
             "18A 项目核心是技术管线, 业务质量权重应上调 +10pp"),
            ("估值合理性", -0.05, -0.05,
             "18A 估值无法用 PE/PS, 主要靠 rNPV/Peak Sales, 估值锚定可靠度低"),
            ("风控等级", +0.05, +0.05,
             "18A 临床失败 / 专利 / 监管风险显著高于通用项目"),
        ]
    # ---- 18C 特专科技 ----
    elif profile.listing_chapter == "Main_Board_18C":
        if profile.profitability_stage == "Pre_Commercial":
            adjustments += [
                ("业务质量", +0.15, +0.15,
                 "18C 未商业化项目本质是技术押注, 业务质量权重大幅上调 +15pp"),
                ("估值合理性", -0.10, -0.10,
                 "18C 未商业化无收入, 估值方法权重应降低（仅作 sanity check）"),
                ("宏观与行业窗口", +0.05, +0.05,
                 "18C 项目对赛道 / 政策窗口尤为敏感"),
            ]
        else:
            adjustments += [
                ("业务质量", +0.05, +0.05,
                 "18C 已商业化项目, 业务可量化, 微调 +5pp"),
                ("估值合理性", -0.05, -0.05,
                 "PE 不适用, 用 PS/EV-Sales, 锚定可靠度略降"),
            ]

    # ---- AH 双重 / A 股已上市 ----
    if profile.needs_a_share_anchor:
        adjustments += [
            ("估值合理性", -0.05, +0.00,
             "AH 双重 估值简化为 A 股 × 折价, 方法论复杂度低, 可降下沿"),
            ("情绪与流动性", +0.05, +0.05,
             "A-H 联动 + 港股流动性差异对短期情绪影响显著"),
        ]

    # ---- 二次上市（中概回归等）----
    if profile.is_secondary_listing:
        adjustments += [
            ("情绪与流动性", +0.05, +0.10,
             "主上市市场（如美股）波动会快速传导, 情绪权重上调"),
            ("宏观与行业窗口", +0.05, +0.05,
             "中美博弈 / SEC 监管对二次上市影响突出"),
        ]

    # ---- WVR 同股不同权 ----
    if profile.has_wvr:
        adjustments.append(
            ("风控等级", +0.05, +0.05,
             "同股不同权架构下, 投票权稀释 / 创始人控制风险显著, 风控权重 +5pp")
        )

    # ---- 规模档 ----
    if profile.size_tier == "Small":
        adjustments += [
            ("情绪与流动性", +0.05, +0.10,
             "小盘股流动性风险显著, 情绪权重上调"),
            ("辩论倾向", -0.05, -0.05,
             "小盘股研报覆盖少, 辩论质量受信息约束, 权重略降"),
        ]
    elif profile.size_tier == "Mega":
        adjustments += [
            ("情绪与流动性", -0.05, -0.05,
             "巨盘股流动性充分, 情绪短期权重应降低"),
            ("辩论倾向", -0.03, -0.03,
             "巨盘股共识高, 辩论分歧少, 权重略降"),
        ]

    # ---- 行业主题 ----
    if profile.industry_theme == "Bio_Pharma":
        adjustments += [
            ("业务质量", +0.05, +0.05,
             "生物医药项目, 管线 / 研发能力是核心, 业务权重 +5pp"),
        ]
    elif profile.industry_theme in ("Real_Estate", "Financial"):
        adjustments += [
            ("宏观与行业窗口", +0.05, +0.10,
             "地产 / 金融对宏观利率 / 政策周期敏感, 宏观权重应上调"),
            ("业务质量", -0.05, -0.05,
             "地产 / 金融受宏观决定大于个体, 业务质量权重略降"),
        ]

    # ---- 应用调整 ----
    # 同因子的多次调整累加
    factor_to_total: dict[str, list[float]] = {}  # factor → [lower_delta, upper_delta]
    factor_to_reasons: dict[str, list[str]] = {}
    for factor, low_delta, up_delta, reason in adjustments:
        if factor not in factor_to_total:
            factor_to_total[factor] = [0.0, 0.0]
            factor_to_reasons[factor] = []
        factor_to_total[factor][0] += low_delta
        factor_to_total[factor][1] += up_delta
        factor_to_reasons[factor].append(reason)

    for f in base:
        name = f["factor"]
        if name in factor_to_total:
            low_delta, up_delta = factor_to_total[name]
            base_low, base_up = f["suggested_weight_range"]
            # clamp 到 [0, 1]
            new_low = max(0.0, min(1.0, base_low + low_delta))
            new_up = max(new_low, min(1.0, base_up + up_delta))
            f["suggested_weight_range"] = (round(new_low, 3), round(new_up, 3))
            f["profile_adjustments"] = factor_to_reasons[name]
        else:
            f["profile_adjustments"] = []

    return base


# ---------------------------- 估值方法硬约束 ----------------------------

def recommended_valuation_methods(profile: ListingProfile) -> dict[str, Any]:
    """根据 profile 推荐 / 禁用的估值方法."""
    if profile.listing_chapter == "Main_Board_18A":
        return {
            "primary": ["rNPV (管线风险调整 NPV)", "Peak Sales × 倍数 × 上市概率",
                        "Pipeline Comparable (临床阶段对标)"],
            "secondary": ["PB (现金折价 sanity check)"],
            "forbidden": ["PE", "PEG", "EV/EBITDA"],
            "rationale": "18A 公司未盈利, PE/EV-EBITDA 不适用; 主战场是临床管线估值",
        }

    if profile.listing_chapter == "Main_Board_18C":
        if profile.profitability_stage == "Pre_Commercial":
            return {
                "primary": ["DCF (5 年现金流贴现)", "EV/Sales (假设 commercialize 后)",
                            "Cost-to-Replicate (技术资产重置成本)"],
                "secondary": ["PB (净资产)"],
                "forbidden": ["PE", "PEG", "EV/EBITDA"],
                "rationale": "18C 未商业化, 营收为零或极低, 用 DCF + 资产法",
            }
        return {
            "primary": ["PS", "EV/Sales", "DCF"],
            "secondary": ["PE (若已盈利)", "PEG (若已盈利且增速可量化)"],
            "forbidden": [],
            "rationale": "18C 已商业化项目, PS / EV-Sales 主导, PE 视盈利情况",
        }

    if profile.needs_a_share_anchor:
        return {
            "primary": ["A-H Discount Anchor (A 股市值 × 0.6-0.85 港股折价)"],
            "secondary": ["Comparable HK Peer (参考但不主导)",
                          "PE / PS (与 A 股对照)"],
            "forbidden": [],
            "rationale": "AH 双重的港股估值锚定 A 股, 历史折价 15-40% 是主要参数",
        }

    if profile.is_secondary_listing:
        return {
            "primary": [f"主上市市场（{profile.main_listing_market or '美股'}）市值 × 港股折价"],
            "secondary": ["Comparable HK Peer", "PE/PS"],
            "forbidden": [],
            "rationale": "二次上市估值跟随主上市, 港股是衍生定价",
        }

    # ---- 通用按盈利状态 ----
    if profile.profitability_stage in ("Profitable_Stable", "Profitable_Growth"):
        return {
            "primary": ["PE", "PEG", "PS"],
            "secondary": ["EV/EBITDA", "DCF"],
            "forbidden": [],
            "rationale": "已盈利项目, PE/PEG 适用",
        }
    if profile.profitability_stage in ("Loss_Making_Growth", "Pre_Profit_Late_Stage"):
        return {
            "primary": ["PS", "EV/Sales", "PB"],
            "secondary": ["DCF (sanity check)"],
            "forbidden": ["PE", "PEG", "EV/EBITDA"],
            "rationale": "亏损成长公司, PE 失效, 用 PS 主导",
        }

    return {
        "primary": ["PS", "EV/Sales"],
        "secondary": ["PE (若适用)"],
        "forbidden": [],
        "rationale": "档案信息不足, 默认 PS 主导",
    }


# ---------------------------- 风险维度增补 ----------------------------

def extra_risk_dimensions(profile: ListingProfile) -> list[dict[str, str]]:
    """根据 profile 返回应额外纳入风控评估的风险维度.

    每条: {dimension: 维度名, why: 为什么这个 profile 需要}
    """
    out: list[dict[str, str]] = []

    if profile.listing_chapter == "Main_Board_18A":
        out += [
            {"dimension": "临床试验失败风险",
             "why": "18A 公司核心产品仍在临床, II/III 期失败概率不可忽视 (历史 II→III 失败率 ~50%)"},
            {"dimension": "专利与商业化时间窗",
             "why": "专利到期前能否商业化 + 海外授权落地是估值关键"},
            {"dimension": "监管审批进度风险",
             "why": "NMPA / FDA 审批周期不可控, 推迟会显著影响 NPV"},
        ]

    if profile.listing_chapter == "Main_Board_18C":
        out += [
            {"dimension": "技术商业化进度风险",
             "why": "18C 项目技术能否落地为可销售产品是核心不确定性"},
            {"dimension": "技术路线被颠覆风险",
             "why": "特专科技赛道（AI / 半导体 / 新能源）技术演进快, 路线选择错误可能整盘归零"},
        ]

    if profile.has_wvr:
        out += [
            {"dimension": "投票权稀释风险",
             "why": "WVR 架构下普通股东对重大决策影响有限, 创始人决策失误难以制衡"},
            {"dimension": "创始人锁定与一致行动协议",
             "why": "WVR 公司应特别关注创始人锁定期 / 退出机制"},
        ]

    if profile.needs_a_share_anchor:
        out += [
            {"dimension": "A-H 折价收敛/扩大风险",
             "why": "AH 双重的港股价格波动很大程度由 A-H 折价决定, 历史折价区间 [-10%, +50%]"},
            {"dimension": "A 股流动性与监管联动",
             "why": "A 股停牌 / 减持新规会立刻传导到港股端"},
        ]

    if profile.is_secondary_listing:
        out += [
            {"dimension": "主上市地波动传导",
             "why": f"主上市在 {profile.main_listing_market or '美股'}, 美股暴跌会次日传导到港股盘"},
            {"dimension": "中概股退市 / SEC 监管风险",
             "why": "PCAOB 审计 / 数据出境 / 中美关系不确定性"},
        ]

    if profile.size_tier == "Small":
        out.append({
            "dimension": "流动性枯竭风险",
            "why": "小盘股 (<30 亿) 上市后日均成交可能跌破 1000 万 HKD, 影响基石解禁退出",
        })

    return out


# ---------------------------- prompt 渲染 ----------------------------

def render_profile_for_prompt(profile: ListingProfile | None) -> str:
    """把 ListingProfile 渲染成 Decision/Comparable/Risk Agent 共用的 prompt 块."""
    if profile is None or profile.listing_chapter == "Unknown":
        return ""

    val_methods = recommended_valuation_methods(profile)
    weight_ranges = adjusted_weight_ranges(profile)
    extra_risks = extra_risk_dimensions(profile)

    lines = [
        "# 上市档案 (Listing Profile) ⭐",
        f"- 上市规则章节: **{profile.listing_chapter}**",
        f"- 盈利阶段: {profile.profitability_stage}",
        f"- 规模档: {profile.size_tier}",
        f"- 行业主题: {profile.industry_theme}",
    ]
    if profile.has_wvr:
        lines.append("- ⚠ 同股不同权 (WVR): 是")
    if profile.has_a_share_listed:
        lines.append(
            f"- ⚠ 已 A 股上市: 是 (代码 {profile.a_share_ticker or '未知'})"
        )
    if profile.is_secondary_listing:
        lines.append(f"- ⚠ 二次上市, 主上市地: {profile.main_listing_market or '未知'}")
    lines.append(f"- 档案推断置信度: {profile.detection_confidence}")

    # 估值方法
    lines += ["", "## 推荐估值方法 (硬约束)", ""]
    if val_methods.get("primary"):
        lines.append(f"**主用**: {', '.join(val_methods['primary'])}")
    if val_methods.get("secondary"):
        lines.append(f"次要 (sanity check): {', '.join(val_methods['secondary'])}")
    if val_methods.get("forbidden"):
        lines.append(f"**禁用**: {', '.join(val_methods['forbidden'])} (该 profile 下不适用)")
    lines.append(f"理由: {val_methods.get('rationale', '')}")

    # 调整后权重区间
    adjusted = [f for f in weight_ranges if f.get("profile_adjustments")]
    if adjusted:
        lines += ["", "## 因子权重区间调整 (相对 STANDARD_DECISION_FACTORS)", ""]
        for f in adjusted:
            new_low, new_up = f["suggested_weight_range"]
            lines.append(
                f"- **{f['factor']}**: 区间调整为 [{new_low:.0%}, {new_up:.0%}]"
            )
            for reason in f["profile_adjustments"]:
                lines.append(f"  - {reason}")

    # 额外风险维度
    if extra_risks:
        lines += ["", "## 应额外纳入风控评估的风险维度", ""]
        for r in extra_risks:
            lines.append(f"- **{r['dimension']}**: {r['why']}")

    return "\n".join(lines) + "\n"
