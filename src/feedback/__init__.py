"""反馈循环（闭环学习）模块。

包含三大组件：
- models  : Prediction / Outcome / ScoreCards 等 Pydantic 模型
- store   : SQLite 持久化层（投决归档 + 结果跟踪）
- cli     : record-outcome / list / export 命令（在主 cli.py 中注册）

设计原则：
1. 评分卡是 Phase A 的关键——所有未来的统计/校准/复盘都依赖这层结构化数据。
2. Outcome 字段紧贴港股 IPO 基石场景（6 个月禁售期、破发率、流动性）。
3. SQLite 而非 JSON 文件：天然支持查询/聚合/JOIN，单机够用，零运维。
"""
from .models import (
    AgentScoreCard,
    ProspectusScoreCard,
    IndustryScoreCard,
    MacroScoreCard,
    ComparableScoreCard,
    TechTrendScoreCard,
    SentimentScoreCard,
    RiskScoreCard,
    Prediction,
    Outcome,
    Score,
)
from .store import FeedbackStore

__all__ = [
    "AgentScoreCard",
    "ProspectusScoreCard",
    "IndustryScoreCard",
    "MacroScoreCard",
    "ComparableScoreCard",
    "TechTrendScoreCard",
    "SentimentScoreCard",
    "RiskScoreCard",
    "Prediction",
    "Outcome",
    "Score",
    "FeedbackStore",
]
