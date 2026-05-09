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


def test_ths_basic_data_parser_column_format() -> None:
    from src.data.ths_client import _parse_table_to_dict_by_code

    payload = {
        "tables": {
            "thscode": ["00700.HK", "09988.HK"],
            "ths_corp_chi_name_stock": ["腾讯控股", "阿里巴巴"],
            "ths_main_business_stock": ["互联网增值服务", "电商及云计算"],
        }
    }
    out = _parse_table_to_dict_by_code(
        payload, ["00700.HK", "09988.HK"], ["ths_corp_chi_name_stock", "ths_main_business_stock"]
    )
    assert out["00700.HK"]["ths_corp_chi_name_stock"] == "腾讯控股"
    assert out["09988.HK"]["ths_main_business_stock"] == "电商及云计算"


def test_ths_edb_parser_column_format() -> None:
    from src.data.ths_client import _parse_edb_payload

    payload = {
        "tables": {
            "time": ["2025-01-31", "2025-02-28", "2025-03-31"],
            "M002820027": [4.5, 4.4, 4.3],
            "M002824001": [7.78, 7.79, 7.80],
        }
    }
    out = _parse_edb_payload(payload, {"HIBOR_1M": "M002820027", "USD_HKD": "M002824001"})
    assert len(out["HIBOR_1M"]) == 3
    assert out["HIBOR_1M"][-1]["value"] == 4.3
    assert out["USD_HKD"][0]["date"] == "2025-01-31"


def test_rag_bge_query_prefix_detection() -> None:
    from src.data.rag import ProspectusRAG

    rag_bge = ProspectusRAG(project_id="t", embedding_model="BAAI/bge-base-zh-v1.5")
    rag_minilm = ProspectusRAG(project_id="t", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert rag_bge._is_bge_zh() is True
    assert rag_minilm._is_bge_zh() is False


def test_router_provider_specific_override(monkeypatch) -> None:
    """provider-specific env > generic env > default."""
    from config import get_settings
    from src.llm.router import ModelTier, resolve_model

    get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("MODEL_TIER_DECIDE_KIMI", "kimi-latest-overridden")
    assert resolve_model(ModelTier.DECIDE) == "kimi-latest-overridden"

    get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("MODEL_TIER_DECIDE_KIMI", raising=False)
    # 不设 deepseek override，应走内置默认
    assert resolve_model(ModelTier.DECIDE) == "deepseek-reasoner"

    get_settings.cache_clear()


def test_llm_client_with_mock_provider() -> None:
    """用 mock provider 验证 LLMClient → ledger 链路。"""
    from src.llm import LLMClient
    from src.llm.client import LLMResponse
    from src.llm.router import ModelTier

    class _Mock:
        def complete(self, *, model, system, messages, max_tokens, temperature, cached_system_blocks):
            return LLMResponse(
                text="mock 响应",
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=20,
                model=model,
            )

    client = LLMClient(provider=_Mock(), provider_name="mock")
    resp = client.complete(
        tier=ModelTier.ANALYZE,
        system="你是测试 agent",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.text == "mock 响应"
    assert client.ledger.by_tier["analyze"]["input"] == 100
    assert client.ledger.by_tier["analyze"]["calls"] == 1


def test_workflow_extras_typed_access() -> None:
    """WorkflowExtras 已知字段属性访问，未知字段 fallback 到 misc。"""
    from src.agents.extras import WorkflowExtras

    ex = WorkflowExtras()
    ex.peer_pe_multiples = [10.0, 12.0, 15.0]
    ex.set("custom_field", "hello")

    assert ex.peer_pe_multiples == [10.0, 12.0, 15.0]
    assert ex.get("peer_pe_multiples") == [10.0, 12.0, 15.0]
    assert ex.get("nonexistent_key", default="dft") == "dft"
    assert ex.get("custom_field") == "hello"
    assert ex.misc["custom_field"] == "hello"


def test_workflow_extras_from_dict_routes_unknown_to_misc() -> None:
    from src.agents.extras import WorkflowExtras

    ex = WorkflowExtras.from_dict({
        "company_basic": {"name": "test"},
        "industry_research": [{"title": "r1"}],
        "totally_unknown_key": [1, 2, 3],
    })
    assert ex.company_basic == {"name": "test"}
    assert ex.industry_research == [{"title": "r1"}]
    assert ex.misc["totally_unknown_key"] == [1, 2, 3]


def test_openai_compat_provider_merges_system_into_messages() -> None:
    """验证 _OpenAICompatProvider 把 system + cached_system_blocks 合并到 messages[0]."""
    from src.llm.client import _OpenAICompatProvider, LLMResponse

    class _FakeOpenAI:
        def __init__(self):
            self.last_call = None
            self.chat = self
            self.completions = self

        def create(self, *, model, messages, max_tokens, temperature):
            self.last_call = {"model": model, "messages": messages}
            class _Choice:
                message = type("M", (), {"content": "ok"})()
            class _Usage:
                prompt_tokens = 10
                completion_tokens = 5
                prompt_cache_hit_tokens = 0
            return type("R", (), {"choices": [_Choice()], "usage": _Usage()})()

    p = _OpenAICompatProvider.__new__(_OpenAICompatProvider)
    p.client = _FakeOpenAI()
    p.provider_name = "test"

    p.complete(
        model="kimi-latest",
        system="你是分析师",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.3,
        cached_system_blocks=["招股书静态内容"],
    )
    msgs = p.client.last_call["messages"]
    assert msgs[0]["role"] == "system"
    # cached blocks 在前，system 在后
    assert msgs[0]["content"].startswith("招股书静态内容")
    assert "你是分析师" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
