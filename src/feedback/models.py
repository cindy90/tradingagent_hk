"""反馈循环数据模型。

ScoreCard 设计要点:
- 每个分析 Agent 有专属 ScoreCard（字段贴合该 Agent 的分析维度）
- 所有 ScoreCard 都继承 AgentScoreCard 公共字段（summary/scores/evidence）
- 数值字段尽量 0-5 标度（直观 + 便于跨 Agent 加权）
- evidence_pages 记录招股书页码，便于复盘时核查

未来 Phase D 启动数据驱动校准时，所有这些字段都是 feature 输入。
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------- 公共基类 ----------

class AgentScoreCard(BaseModel):
    """所有 Agent 评分卡的公共字段。"""
    model_config = ConfigDict(extra="allow")

    summary: str = Field(description="一句话核心结论 (≤80 字)")
    overall_score: float = Field(ge=0, le=5, description="该 Agent 视角下的综合评分 0-5")
    confidence: Literal["高", "中", "低"] = "中"
    evidence_pages: list[int] = Field(default_factory=list, description="招股书页码引用")
    notes: str = ""

    @field_validator("evidence_pages", mode="before")
    @classmethod
    def _coerce_evidence_pages(cls, v: Any) -> list[int]:
        # LLM 偶尔会塞章节名（"招股书行业概览章节"）或 "P.42" / "42页" 等字符串。
        # 严格校验会让整张 score card 作废。这里宽容地抽数字，无数字则丢弃该项。
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        out: list[int] = []
        for item in v:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, float):
                out.append(int(item))
            elif isinstance(item, str):
                m = re.search(r"\d+", item)
                if m:
                    out.append(int(m.group()))
        return out


# ---------- 各 Agent 专属评分卡 ----------

class ProspectusScoreCard(AgentScoreCard):
    """招股书深度分析评分卡。"""
    business_quality: float = Field(ge=0, le=5)
    financial_quality: float = Field(ge=0, le=5)
    risk_severity: float = Field(ge=0, le=5, description="0=极低风险 5=极高风险")
    valuation_anchor_basis: float = Field(ge=0, le=5, description="估值锚定可靠度")
    cornerstone_terms_attractiveness: float = Field(ge=0, le=5)

    # 关键数值（让决策可量化）
    revenue_cagr_3y: float | None = None
    net_margin_latest: float | None = None
    gross_margin_latest: float | None = None
    customer_concentration_top5_pct: float | None = None
    related_party_revenue_pct: float | None = None
    rd_intensity_latest: float | None = None


class IndustryScoreCard(AgentScoreCard):
    market_size_cagr: float | None = None
    competition_intensity: float = Field(ge=0, le=5, description="0=分散 5=高度垄断")
    industry_lifecycle_stage: Literal["early", "growth", "mature", "declining", "unknown"] = "unknown"
    industry_score: float = Field(ge=0, le=5)
    company_market_share_pct: float | None = None
    top3_concentration_pct: float | None = None


class MacroScoreCard(AgentScoreCard):
    market_window_score: float = Field(ge=0, le=5, description="0=逆风 5=极佳")
    hibor_trend: Literal["rising", "stable", "falling", "unknown"] = "unknown"
    hsi_pe_percentile: float | None = None  # 0-1
    hk_ipo_break_rate_recent: float | None = None  # 近 3 月港股新股破发率
    northbound_flow_trend: Literal["inflow", "outflow", "neutral", "unknown"] = "unknown"


class ComparableScoreCard(AgentScoreCard):
    peer_count: int = 0
    median_pe: float | None = None
    median_ps: float | None = None
    median_ev_ebitda: float | None = None
    median_peg: float | None = None  # 新增: PEG = PE / 净利润 CAGR
    valuation_low_hkd_b: float | None = None
    valuation_mid_hkd_b: float
    valuation_high_hkd_b: float | None = None
    valuation_method: Literal["PE", "PS", "EV/EBITDA", "DCF", "weighted", "other", "PEG", "SOTP"] = "PE"
    ipo_discount_assumed_pct: float = 0.15  # 港股 IPO 通常打 10-25% 折扣

    # 反推估值: 招股价中枢隐含的未来 1 年营收 CAGR (%)
    implied_revenue_cagr_at_ipo: float | None = None
    # 多业务线 SOTP 分拆 (可选): {业务线名: 估值贡献亿 HKD}
    sotp_breakdown: dict[str, float] = Field(default_factory=dict)


class TechTrendScoreCard(AgentScoreCard):
    moat_strength: float = Field(ge=0, le=5)
    rd_intensity: float | None = None
    tech_lifecycle: Literal["emerging", "growth", "mature", "declining", "unknown"] = "unknown"
    disruption_risk: float = Field(ge=0, le=5, description="被颠覆风险, 0=低 5=高")


class SentimentScoreCard(AgentScoreCard):
    sector_momentum: float = Field(ge=-1, le=1, description="-1=极冷 1=极热")
    similar_ipo_break_rate: float | None = None
    market_attention_score: float = Field(ge=0, le=5)


class ScarcityScoreCard(AgentScoreCard):
    """稀缺性评分卡: 标的在港股同主题中的供给稀缺度。"""
    scarcity_score: float = Field(
        ge=1, le=5,
        description="1=红海/烂大街 / 3=中性 / 5=独苗稀缺. LLM 在引擎 raw_scarcity_score 基础上微调",
    )
    listed_count_in_theme: int = Field(
        default=0, description="同主题港股已上市公司数 (不含 target)",
    )
    market_cap_top3_share: float | None = Field(
        default=None, ge=0, le=1, description="同主题 top 3 市值占比",
    )
    liquidity_thinning_ratio: float | None = Field(
        default=None, ge=0, le=1,
        description="同主题中 30 日均成交 < 5000 万 HKD 的公司占比",
    )
    recent_ipo_count_12m: int = Field(
        default=0, description="过去 12 月同主题 IPO 数 (边际稀缺度)",
    )
    pipeline_count_in_theme: int = Field(
        default=0,
        description="v2 流量稀缺度: 同主题港股 IPO 排队中 (申请版本/已通过聆讯) 公司数",
    )
    pipeline_companies: list[str] = Field(
        default_factory=list,
        description="排队中匹配 target 主题的公司名 (前 5 家足以)",
    )
    sentiment_linkage: Literal[
        "稀缺+热情", "稀缺+冷淡", "拥挤+热情", "拥挤+冷淡", "中性",
    ] = Field(
        default="中性",
        description="稀缺度 × 情绪联动. 稀缺+热情=溢价 / 稀缺+冷淡=错杀机会 / "
                    "拥挤+热情=情绪轮动末端 / 拥挤+冷淡=最弱组合",
    )
    valuation_premium_view: Literal["允许溢价", "中性", "应折扣", "无明确观点"] = "中性"
    differentiator: list[str] = Field(
        default_factory=list,
        description="即使同主题已有多家, target 的差异化点 (技术/规模/客户结构), 1-3 条",
    )


class RiskItem(BaseModel):
    """单维度风险的三维量化（概率 × 影响 × 预警信号）。"""
    model_config = ConfigDict(extra="ignore")
    dimension: str  # 财务造假 / 行业逆风 / 估值高估 / ...
    score: float = Field(ge=1, le=5, description="1=极高 5=极低")
    probability: Literal["极低", "低", "中", "高"] = "中"
    impact: Literal["小", "中", "大", "极大"] = "中"
    early_warning: list[str] = Field(default_factory=list, description="可观测的预警阈值")


class RiskScoreCard(AgentScoreCard):
    overall_risk_level: float = Field(ge=1, le=5, description="1=极高风险 5=极低")
    risk_dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="8 维风险综合评分（向后兼容）: 财务造假/行业逆风/估值高估/流动性/股东减持/监管/治理/ESG",
    )
    # 新增: 三维量化的详细风险项
    detailed_risks: list[RiskItem] = Field(
        default_factory=list,
        description="每维度风险的概率 × 影响 × 预警信号，专业级风险矩阵的结构化输出",
    )
    veto_conditions: list[str] = Field(default_factory=list)
    must_satisfy_conditions: list[str] = Field(default_factory=list)


SCORE_CARD_CLASSES: dict[str, type[AgentScoreCard]] = {
    "prospectus_analyst": ProspectusScoreCard,
    "industry": IndustryScoreCard,
    "macro": MacroScoreCard,
    "comparable": ComparableScoreCard,
    "tech_trend": TechTrendScoreCard,
    "sentiment": SentimentScoreCard,
    "scarcity": ScarcityScoreCard,
    "risk": RiskScoreCard,
}


# ---------- 决策落库 (Prediction) ----------

class Prediction(BaseModel):
    """一次完整的投决记录。落 SQLite predictions 表。"""
    model_config = ConfigDict(extra="ignore")

    project_id: str
    ticker: str
    company_name: str
    industry: str
    decision_date: datetime

    # 决议核心字段（来自 DecisionResult）
    recommendation: str
    confidence: str
    valuation_low: float | None = None
    valuation_mid: float
    valuation_high: float | None = None
    anchor_method: str
    anchor_logic: str = ""
    ipo_pricing_view: str
    suggested_amount_low_usd_m: float
    suggested_amount_high_usd_m: float
    key_supports: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    deal_conditions: list[str] = Field(default_factory=list)
    monitoring_kpis: list[str] = Field(default_factory=list)

    # 各 Agent 评分卡（structured，便于回溯和统计）
    agent_score_cards: dict[str, dict] = Field(default_factory=dict)

    # 评审分（Phase B）
    reviewer_scores: dict[str, float] = Field(default_factory=dict)

    # 多样性 variants 使用情况（Phase B）
    diversity_variants: dict[str, list[str]] = Field(default_factory=dict)

    # 决策因子加权打分卡 v4 (落库, 供 PostmortemAgent 读取做权重校准)
    decision_weights: list[dict] = Field(
        default_factory=list,
        description="当时的决策因子加权打分卡, 每项含 factor / weight / score / contribution / "
        "source_agents / rationale. 用于 PostmortemAgent 事后做权重校准.",
    )
    weighted_total_score: float | None = None
    weighted_to_recommendation_mapping: str = ""

    # 上市档案核心字段（v5: 用于 calibration priors 三维查询）
    listing_chapter: str = Field(
        default="Unknown",
        description="HKEX 上市规则章节 (Main_Board_Standard / 18A / 18C / 19C / "
        "Secondary_Listing / Dual_Primary_AH / GEM / Unknown)",
    )
    size_tier: str = Field(
        default="Unknown",
        description="规模档 (Small / Mid / Large / Mega / Unknown)",
    )
    industry_theme: str = Field(default="Other", description="行业主题大类")
    has_wvr: bool = False
    has_a_share_listed: bool = False

    # 可复现性元信息
    model_provider: str
    model_tier_models: dict[str, str] = Field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    estimated_cost_cny: float | None = None
    cogalpha_features_used: list[str] = Field(default_factory=list)

    # 文件路径（便于回溯完整 markdown 报告）
    reports_dir_path: str = ""

    # 状态
    status: Literal["open", "closed", "abandoned"] = "open"


# ---------- 实际结果 (Outcome) ----------

class Outcome(BaseModel):
    """投决后的实际结果。一个 prediction 可对应一条 outcome（最新）或多条（每次更新）。"""
    model_config = ConfigDict(extra="ignore")

    prediction_id: int  # FK
    recorded_date: datetime

    # IPO 招股结果
    ipo_actual_price_hkd: float | None = None
    ipo_actual_marketcap_hkd_billion: float | None = None
    final_listing_date: date | None = None

    # 标准 horizon 收益率（相对 IPO 招股价）
    d1_return: float | None = None
    d30_return: float | None = None
    d90_return: float | None = None
    d180_return: float | None = None  # ⭐ 6 个月禁售期满
    d365_return: float | None = None

    # 相对收益（vs 恒指）
    d180_alpha_vs_hsi: float | None = None
    d365_alpha_vs_hsi: float | None = None

    # 锁定期专属
    was_broken_ipo_d1: bool | None = None  # 首日是否破发
    was_broken_ipo_d180: bool | None = None  # 6 月内是否破发
    max_drawdown_in_lockup_pct: float | None = None
    min_price_in_lockup_hkd: float | None = None

    # 流动性
    avg_daily_turnover_hkd_m_d180: float | None = None
    free_float_pct: float | None = None

    # 基石专属
    cornerstone_actual_amount_usd_million: float | None = None
    cornerstone_lockup_end_date: date | None = None
    cornerstone_realized_return_pct: float | None = None  # 解禁日 vs 招股价

    # 自由文本
    user_notes: str = ""
    notable_events: list[str] = Field(default_factory=list)


# ---------- 评分对比 (Score) ----------

class WeightCalibration(BaseModel):
    """单个决策因子的事后权重校准建议。

    PostmortemAgent 在复盘时, 对当时的 decision_weights 每个因子输出一条:
    "事后看, 这个因子的权重应该调到多少, 为什么"。

    多个项目积累后, 系统按 (industry, recommendation) 聚合这些 calibration
    作为下个相似项目的 prior, 注入 Decision Agent prompt.
    """
    model_config = ConfigDict(extra="ignore")

    factor: str = Field(description="因子名 (与 decision_weights 中的 factor 对应)")
    actual_weight_used: float = Field(ge=0, le=1, description="当时给的权重")
    suggested_weight: float = Field(ge=0, le=1, description="事后看应该给的权重")
    delta: float = Field(default=0.0, description="suggested - actual, 自动算")
    rationale: str = Field(description="为什么调整 (基于实际结果)")

    @model_validator(mode="after")
    def _ensure_delta(self) -> "WeightCalibration":
        # delta 缺失或为 0 时自动算 suggested - actual
        if self.delta == 0.0:
            object.__setattr__(
                self, "delta",
                round(self.suggested_weight - self.actual_weight_used, 4),
            )
        return self


class Score(BaseModel):
    """Postmortem Agent 输出的复盘评分。Phase C 用。"""
    model_config = ConfigDict(extra="ignore")

    prediction_id: int
    score_date: datetime

    # 决议正确度（综合）
    recommendation_score: float = Field(ge=-1, le=1, description="-1=完全错 0=中性 1=完全对")

    # 估值精度
    valuation_within_range: bool | None = None
    valuation_error_pct: float | None = None  # (实际 - 中枢) / 中枢

    # 风险识别命中率
    risks_total_count: int = 0
    risks_realized_count: int = 0
    unforeseen_risks_count: int = 0  # 实际发生但我们没识别的

    # Per-Agent 质量评分
    per_agent_quality: dict[str, float] = Field(default_factory=dict)

    # 置信度校准
    confidence_calibration_delta: float | None = None  # 高置信度 - 实际命中率

    # 复盘备忘录
    postmortem_memo: str = ""
    error_root_causes: list[str] = Field(default_factory=list)  # 信息缺失/推理错误/估值不当/黑天鹅

    # 决策因子权重校准建议（Phase A 新增, 供 Phase B 聚合用）
    weight_calibrations: list[WeightCalibration] = Field(
        default_factory=list,
        description="每个决策因子的事后权重校准建议. 累积多个项目后, "
        "按 (industry, recommendation) 聚合作为新项目的 prior",
    )
