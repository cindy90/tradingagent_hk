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


def test_select_cached_blocks_picks_key_sections() -> None:
    from src.data.prospectus import ProspectusChunk
    from src.graph.workflow import select_cached_blocks

    chunks = [
        ProspectusChunk(chunk_id="1", text="封面 cover", page_start=1, page_end=1, section="封面"),
        ProspectusChunk(chunk_id="2", text="目录 toc", page_start=2, page_end=3, section="目录"),
        ProspectusChunk(chunk_id="3", text="承销商列表", page_start=4, page_end=4, section="承销商"),
        ProspectusChunk(chunk_id="4", text="本招股书概要内容...", page_start=5, page_end=20, section="概要"),
        ProspectusChunk(chunk_id="5", text="风险因素详细列举...", page_start=21, page_end=40, section="风险因素"),
        ProspectusChunk(chunk_id="6", text="业务模式与产品线...", page_start=41, page_end=80, section="业务"),
        ProspectusChunk(chunk_id="7", text="财务三表...", page_start=81, page_end=100, section="财务资料"),
    ]
    selected = select_cached_blocks(chunks)
    # 关键章节应被命中，封面/目录/承销商应被跳过
    joined = "\n".join(selected)
    assert "概要" in joined
    assert "风险因素" in joined
    assert "业务" in joined
    assert "封面" not in joined
    assert "目录" not in joined


def test_select_cached_blocks_falls_back_when_no_match() -> None:
    """关键章节都没识别出来时退化为前 3 块。"""
    from src.data.prospectus import ProspectusChunk
    from src.graph.workflow import select_cached_blocks

    chunks = [
        ProspectusChunk(chunk_id=str(i), text=f"chunk{i}", page_start=i, page_end=i,
                        section="无章节标题")
        for i in range(1, 6)
    ]
    selected = select_cached_blocks(chunks)
    assert len(selected) == 3
    assert selected[0] == "chunk1"


def test_select_cached_blocks_respects_size_caps() -> None:
    from src.data.prospectus import ProspectusChunk
    from src.graph.workflow import select_cached_blocks

    long_text = "x" * 10000
    chunks = [
        ProspectusChunk(chunk_id="1", text=long_text, page_start=1, page_end=1, section="概要"),
        ProspectusChunk(chunk_id="2", text=long_text, page_start=2, page_end=2, section="风险因素"),
        ProspectusChunk(chunk_id="3", text=long_text, page_start=3, page_end=3, section="业务"),
    ]
    selected = select_cached_blocks(chunks, max_chars_per_block=2000, max_total_chars=5000)
    for s in selected:
        assert len(s) <= 2000
    assert sum(len(s) for s in selected) <= 5000


def test_decision_schema_validation_passes_on_complete_json() -> None:
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "认购",
        "confidence": "中",
        "suggested_amount_usd_million": [10.0, 20.0],
        "valuation_range_hkd_billion": {
            "low": 50.0, "mid": 70.0, "high": 90.0,
            "anchor_method": "PE", "anchor_logic": "对标可比公司中位数 25x",
        },
        "ipo_pricing_view": "合理",
        "key_supports": ["技术领先", "客户集中度下降"],
        "key_risks": ["毛利率波动"],
        "deal_conditions": [],
        "monitoring_kpis": ["季度营收"],
    }
    result, err = validate_decision(parsed)
    assert result is not None
    assert err == ""
    assert result.recommendation == "认购"


def test_decision_schema_validation_fails_on_missing_field() -> None:
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "认购",
        "confidence": "中",
        # 缺 suggested_amount_usd_million / valuation_range_hkd_billion
        "ipo_pricing_view": "合理",
        "key_supports": ["x"],
        "key_risks": ["y"],
    }
    result, err = validate_decision(parsed)
    assert result is None
    assert "suggested_amount_usd_million" in err or "valuation_range_hkd_billion" in err


def test_decision_schema_validation_fails_on_wrong_array_length() -> None:
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "认购",
        "confidence": "中",
        "suggested_amount_usd_million": [10.0],  # ← 应为 2 个
        "valuation_range_hkd_billion": {
            "mid": 70.0, "anchor_method": "PE", "anchor_logic": "x",
        },
        "ipo_pricing_view": "合理",
        "key_supports": ["x"],
        "key_risks": ["y"],
    }
    result, err = validate_decision(parsed)
    assert result is None


def test_decision_agent_retries_on_invalid_schema() -> None:
    """验证 DecisionAgent 在第一次 schema 校验失败后会触发重试。"""
    from src.agents.base import AgentContext
    from src.agents.decision import DecisionAgent
    from src.agents.extras import WorkflowExtras
    from src.llm import LLMClient
    from src.llm.client import LLMResponse
    from src.llm.router import ModelTier
    from pathlib import Path
    import tempfile

    INVALID = '```json\n{"recommendation": "认购"}\n```\n论述: ...'
    VALID = (
        '```json\n'
        '{"recommendation": "审慎参与", "confidence": "中",'
        ' "suggested_amount_usd_million": [10, 20],'
        ' "valuation_range_hkd_billion": {"mid": 70, "anchor_method": "PE", "anchor_logic": "x"},'
        ' "ipo_pricing_view": "合理",'
        ' "key_supports": ["a"], "key_risks": ["b"],'
        ' "deal_conditions": [], "monitoring_kpis": []}\n'
        '```\n论述: ...'
    )

    class _Provider:
        def __init__(self):
            self.calls = 0

        def complete(self, *, model, system, messages, max_tokens, temperature, cached_system_blocks):
            self.calls += 1
            text = INVALID if self.calls == 1 else VALID
            return LLMResponse(text=text, input_tokens=10, output_tokens=10, model=model)

    p = _Provider()
    llm = LLMClient(provider=p, provider_name="mock")
    agent = DecisionAgent(llm)

    with tempfile.TemporaryDirectory() as d:
        ctx = AgentContext(
            project_id="t", ticker="00000", company_name="测试", industry="x",
            reports_dir=Path(d), rag=None, extras=WorkflowExtras(),
        )
        ctx.briefs["macro"] = "宏观 brief"
        report = agent.run(ctx)

    assert p.calls == 2
    assert report.metadata["validated"] is True
    assert ctx.extras.decision_json["recommendation"] == "审慎参与"


def test_fatal_flag_default_and_subclasses() -> None:
    from src.agents.decision import DecisionAgent
    from src.agents.industry import IndustryAgent
    from src.agents.prospectus_analyst import ProspectusAnalystAgent

    assert ProspectusAnalystAgent.fatal is True
    assert DecisionAgent.fatal is True
    assert IndustryAgent.fatal is False  # 默认非致命


def test_non_fatal_failure_injects_failure_brief(tmp_path) -> None:
    """非致命 Agent 失败时，应在 ctx.briefs 写入显式"信息缺失"标记，
    供下游 LLM 感知；不抛异常。"""
    from pathlib import Path
    from src.agents.base import AgentContext, BaseAgent, AgentReport
    from src.agents.extras import WorkflowExtras
    from src.graph.workflow import CornerstoneWorkflow
    from src.llm import LLMClient
    from src.llm.client import LLMResponse
    from src.llm.router import ModelTier

    class _MockProvider:
        def complete(self, *, model, system, messages, max_tokens, temperature, cached_system_blocks):
            return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model=model)

    class _FailingAgent(BaseAgent):
        name = "failing"
        tier = ModelTier.ANALYZE
        description = "一个会失败的 Agent"
        fatal = False

        def run(self, ctx):
            raise RuntimeError("simulated boom")

    class _OkAgent(BaseAgent):
        name = "ok"
        tier = ModelTier.ANALYZE
        description = "一个成功的 Agent"

        def run(self, ctx):
            ctx.briefs["ok"] = "ok brief"
            return AgentReport(agent="ok", full_report="ok full", brief="ok brief")

    llm = LLMClient(provider=_MockProvider(), provider_name="mock")
    wf = CornerstoneWorkflow.__new__(CornerstoneWorkflow)
    wf.llm = llm
    wf.steps = [_FailingAgent(llm), _OkAgent(llm)]

    ctx = AgentContext(
        project_id="t",
        ticker="00000",
        company_name="测试",
        industry="test",
        reports_dir=tmp_path,
        rag=None,
        extras=WorkflowExtras(),
    )

    for i, agent in enumerate(wf.steps, start=1):
        try:
            r = agent.run(ctx)
            ctx.briefs[agent.name] = r.brief
        except Exception as e:
            if agent.fatal:
                raise
            ctx.briefs[agent.name] = f"**[执行失败]** {type(e).__name__}"

    assert "执行失败" in ctx.briefs["failing"]
    assert ctx.briefs["ok"] == "ok brief"


def test_fatal_failure_propagates() -> None:
    from src.agents.base import AgentContext, BaseAgent
    from src.agents.extras import WorkflowExtras
    from pathlib import Path
    import tempfile, pytest

    class _FatalFail(BaseAgent):
        name = "fatal"
        fatal = True

        def run(self, ctx):
            raise RuntimeError("fatal error")

    with tempfile.TemporaryDirectory() as d:
        ctx = AgentContext(
            project_id="t", ticker="00000", company_name="测试", industry="test",
            reports_dir=Path(d), rag=None, extras=WorkflowExtras(),
        )
        agent = _FatalFail.__new__(_FatalFail)
        with pytest.raises(RuntimeError, match="fatal error"):
            agent.run(ctx)


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
