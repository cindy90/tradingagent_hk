"""WorkflowExtras: 跨 Agent 共享数据的强类型容器。

设计要点：
- 已知字段全部声明为属性，IDE/mypy 能 catch typo（之前 ctx.extras.get("peer_pe_multipless") 静默返回空）
- 未知字段走 misc，避免破坏可扩展性
- 保留 .get(key, default) 接口以兼容旧 dict 风格调用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowExtras:
    # 公司层面（来自 THS_BD）
    company_basic: dict = field(default_factory=dict)

    # 宏观/行业（来自 THS_EDB / THS_DR）
    macro_indicators: dict = field(default_factory=dict)
    industry_research: list = field(default_factory=list)

    # 可比公司估值
    peers: list = field(default_factory=list)
    target_net_profit: float | None = None
    target_revenue: float | None = None
    peer_pe_multiples: list = field(default_factory=list)
    peer_ps_multiples: list = field(default_factory=list)

    # 二级市场情绪
    peer_recent_quotes: list = field(default_factory=list)
    recent_hk_ipos: list = field(default_factory=list)
    # peer 近期公告事件（减持 / 业绩预警 / 回购等，影响 sentiment / risk）
    peer_announcements: dict = field(default_factory=dict)  # {ticker: [{date, title, tags}]}

    # 港股三大指数（HSI / HSCEI / HSTECH 实时点位 + PE）, 来自 iFinD
    market_indices: list = field(default_factory=list)

    # 路演阶段手工录入的信号（CLI --dark-pool-price / --oversubscribe-x / --press-coverage 等）
    # iFinD 不一定能拿到, 用户从富途/老虎/媒体补充
    roadshow_signals: dict = field(default_factory=dict)
    # 例如: {"dark_pool_price": 22.5, "oversubscribe_retail_x": 80, "oversubscribe_intl_x": 12,
    #        "press_coverage_score": 4, "press_coverage_notes": "财新/华尔街见闻深度报道 3 篇"}

    # 同期 / 未来 60 天同行业其它 IPO（用于评估资金分流效应）
    competing_ipos: list = field(default_factory=list)

    # 历史权重校准 priors (Phase B): 当样本足够时, workflow 会从 SQLite scores 表
    # 聚合同行业 calibration, 注入 Decision Agent 作为软引导
    weight_priors: dict = field(default_factory=dict)

    # 目标公司本身的 iFinD 估值/财务（IPO 询价阶段也常已建档）
    target_valuation: dict | None = None

    # Agent 间产物
    debate_outcome: Any = None
    decision_json: dict | None = None

    # 未预期键的兜底（不推荐使用，仅为兼容性保留）
    misc: dict = field(default_factory=dict)

    _RESERVED = {"misc", "_RESERVED"}

    def get(self, key: str, default: Any = None) -> Any:
        """兼容 dict 风格的访问。优先返回字段值（None 视为未设），否则查 misc。"""
        if key in self._RESERVED:
            return default
        if hasattr(self, key):
            val = getattr(self, key)
            if val is None or (isinstance(val, (list, dict)) and not val):
                return default if default is not None else val
            return val
        return self.misc.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """显式 setter，未知键写入 misc。"""
        if key in self._RESERVED:
            raise KeyError(f"reserved key: {key}")
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.misc[key] = value

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowExtras":
        """从 dict 构造，未知键自动放进 misc。"""
        known = {f for f in cls.__dataclass_fields__ if f != "misc"}
        kwargs = {k: v for k, v in data.items() if k in known}
        misc = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, misc=misc)
