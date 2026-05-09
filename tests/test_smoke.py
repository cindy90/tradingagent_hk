"""极简 smoke 测试：只验证模块导入和确定性工具函数。

不调用 LLM、不访问网络，CI 友好。完整端到端测试需要真实 API key。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports() -> None:
    from config import get_settings
    from src.llm import LLMClient, ModelTier, resolve_model
    from src.agents import BaseAgent, AgentContext, AgentReport, Summarizer
    from src.graph import CornerstoneWorkflow

    assert ModelTier.ANALYZE.value == "analyze"
    assert isinstance(resolve_model(ModelTier.SUMMARIZE), str)


def test_valuation_tool() -> None:
    from src.tools.valuation import comparable_valuation, percentile

    res = comparable_valuation(
        target_metric=100.0,
        peer_multiples=[10, 12, 15, 18, 20],
        metric_name="PE",
    )
    assert res["peer_count"] == 5
    assert res["multiple_median"] == 15
    assert res["valuation_mid"] == 1500

    assert percentile([1, 2, 3, 4, 5], 0.5) == 3


def test_financials_tool() -> None:
    from src.tools.financials import cagr, summarize_income_trend, safe_div

    assert safe_div(10, 0) is None
    assert safe_div(10, 2) == 5
    assert cagr([100, 121], 1) is not None

    res = summarize_income_trend([
        {"报告期": "2022", "营业收入": 1000, "归属母公司股东净利润": 100},
        {"报告期": "2023", "营业收入": 1500, "归属母公司股东净利润": 200},
        {"报告期": "2024", "营业收入": 2000, "归属母公司股东净利润": 350},
    ])
    assert res["revenue"] == [1000, 1500, 2000]
    assert res["latest_net_margin"] == 0.175


def test_summarizer_system_prompt_has_structure() -> None:
    from src.agents.summarizer import SUMMARIZER_SYSTEM
    assert "核心结论" in SUMMARIZER_SYSTEM
    assert "关键事实" in SUMMARIZER_SYSTEM


def test_decision_json_parser() -> None:
    from src.agents.decision import parse_decision_json

    text = '前置文字\n```json\n{"recommendation": "认购", "confidence": "高"}\n```\n后置文字'
    parsed = parse_decision_json(text)
    assert parsed is not None
    assert parsed["recommendation"] == "认购"
