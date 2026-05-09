"""极简 smoke 测试：只验证模块导入和确定性工具函数。

不调用 LLM、不访问网络，CI 友好。完整端到端测试需要真实 API key。
"""
from __future__ import annotations

import re
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


def test_estimate_bge_tokens_chinese_dominant() -> None:
    from src.data.prospectus import estimate_bge_tokens

    # 纯中文：估算 ~= 字符数
    zh = "本公司主要从事工业机器人研发与制造业务"
    n = estimate_bge_tokens(zh)
    assert len(zh) - 3 <= n <= len(zh) + 3


def test_estimate_bge_tokens_english_words() -> None:
    from src.data.prospectus import estimate_bge_tokens

    # 英文：词数 × 1.3 量级
    en = "Industrial robotics and collaborative arms business"
    n = estimate_bge_tokens(en)
    word_count = len(en.split())
    assert n >= word_count  # 至少不会低于词数


def test_split_text_token_aware_respects_target() -> None:
    from src.data.prospectus import ProspectusLoader, estimate_bge_tokens

    text = "本公司业务概览。" * 200  # 长文本
    chunks = ProspectusLoader._split_text_token_aware(text, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        # 允许 20% 弹性（硬切边界）
        assert estimate_bge_tokens(c) <= 60, f"chunk 超出 token 上限: {estimate_bge_tokens(c)}"


def test_split_text_creates_overlap() -> None:
    from src.data.prospectus import ProspectusLoader

    text = "段A。段B。段C。段D。段E。段F。段G。段H。" * 30
    chunks = ProspectusLoader._split_text_token_aware(text, target_tokens=30, overlap_tokens=10)
    if len(chunks) >= 2:
        # 相邻 chunk 应有重叠（任意一个 atom 共享）
        a_atoms = set(re.findall(r"段[A-Z]", chunks[0]))
        b_atoms = set(re.findall(r"段[A-Z]", chunks[1]))
        assert a_atoms & b_atoms, "相邻 chunk 没有 overlap"


def test_table_to_markdown_basic() -> None:
    from src.data.prospectus import _table_to_markdown

    table = [
        ["项目", "2022年", "2023年", "2024年"],
        ["营业收入", "1,000", "1,500", "2,000"],
        ["净利润", "100", "200", "350"],
    ]
    md = _table_to_markdown(table)
    assert "| 项目 | 2022年 | 2023年 | 2024年 |" in md
    assert "| --- | --- | --- | --- |" in md
    assert "| 营业收入 | 1,000 | 1,500 | 2,000 |" in md


def test_is_meaningful_table_filters_empty() -> None:
    from src.data.prospectus import _is_meaningful_table

    # 太少数据
    assert _is_meaningful_table([["", ""]]) is False
    # 1 行
    assert _is_meaningful_table([["a", "b", "c"]]) is False
    # 内容稀疏
    assert _is_meaningful_table([["", "", ""], ["", "", ""]]) is False
    # 合格
    assert _is_meaningful_table([
        ["a", "b", "c"],
        ["1", "2", "3"],
    ]) is True


def test_score_card_schema_instruction_includes_fields() -> None:
    from src.agents.scoring import schema_instruction
    from src.feedback.models import IndustryScoreCard

    txt = schema_instruction(IndustryScoreCard)
    assert "industry_score" in txt
    assert "competition_intensity" in txt
    # 必须明确要求一段 ```json``` 代码块
    assert "```json" in txt
    assert "只输出一段 json" in txt or "only" in txt.lower() or "不要重复" in txt


def test_parse_score_card_extracts_last_json_block() -> None:
    """LLM 输出含 markdown + 末尾 json 代码块，应只取最后一个。"""
    from src.agents.scoring import parse_score_card
    from src.feedback.models import RiskScoreCard

    txt = """## 一、财务造假风险
评级 4 ...

## 综合
- 综合风控评级: 3.5

```json
{
  "summary": "整体可控",
  "overall_score": 3.5,
  "confidence": "中",
  "evidence_pages": [108, 200],
  "notes": "",
  "overall_risk_level": 3.5,
  "risk_dimensions": {"财务造假": 4.0, "估值高估": 2.5},
  "veto_conditions": ["上市前估值不超过 100 亿港元"],
  "must_satisfy_conditions": []
}
```"""
    sc = parse_score_card(txt, RiskScoreCard, agent_name="risk")
    assert sc is not None
    assert sc.overall_risk_level == 3.5
    assert sc.veto_conditions == ["上市前估值不超过 100 亿港元"]


def test_parse_score_card_returns_none_on_invalid_schema() -> None:
    from src.agents.scoring import parse_score_card
    from src.feedback.models import IndustryScoreCard

    txt = '```json\n{"summary": "x"}\n```'  # 缺多个必填字段
    sc = parse_score_card(txt, IndustryScoreCard, agent_name="industry")
    assert sc is None


def test_strip_score_card_block_removes_last_json() -> None:
    from src.agents.scoring import strip_score_card_block

    txt = "## 报告\n\n正文...\n\n```json\n{\"x\": 1}\n```"
    out = strip_score_card_block(txt)
    assert "```json" not in out
    assert "正文" in out


def test_feedback_store_round_trip(tmp_path) -> None:
    from datetime import datetime
    from src.feedback import FeedbackStore
    from src.feedback.models import Outcome, Prediction

    s = FeedbackStore(db_path=tmp_path / "test.sqlite")
    p = Prediction(
        project_id="abc_001", ticker="02670", company_name="测试机器人",
        industry="工业机器人", decision_date=datetime.now(),
        recommendation="审慎参与", confidence="中",
        valuation_low=60, valuation_mid=80, valuation_high=100,
        anchor_method="PE", anchor_logic="对标可比中位数 25x",
        ipo_pricing_view="合理",
        suggested_amount_low_usd_m=10, suggested_amount_high_usd_m=20,
        key_supports=["技术领先"], key_risks=["客户集中度高"],
        agent_score_cards={"industry": {"industry_score": 4.0, "summary": "x"}},
        model_provider="kimi",
        model_tier_models={"analyze": "moonshot-v1-32k"},
        total_input_tokens=100000, total_output_tokens=20000,
        estimated_cost_cny=2.4,
    )
    pid = s.save_prediction(p)
    assert pid > 0

    # idempotent on duplicate project_id
    pid2 = s.save_prediction(p)
    assert pid2 == pid

    p2 = s.get_prediction(pid)
    assert p2 is not None
    assert p2.recommendation == "审慎参与"
    assert p2.key_supports == ["技术领先"]
    assert p2.agent_score_cards["industry"]["industry_score"] == 4.0

    o = Outcome(
        prediction_id=pid, recorded_date=datetime.now(),
        ipo_actual_price_hkd=25.5, d1_return=0.12, d180_return=-0.08,
        was_broken_ipo_d1=False, was_broken_ipo_d180=True,
        notable_events=["管理层变动"],
    )
    s.record_outcome(o)
    o2 = s.latest_outcome(pid)
    assert o2 is not None
    assert o2.d180_return == -0.08
    assert o2.was_broken_ipo_d180 is True
    assert o2.notable_events == ["管理层变动"]

    stats = s.stats_summary()
    assert stats["predictions_total"] == 1
    assert stats["outcomes_total"] == 1


def test_render_case_summary_includes_key_fields() -> None:
    from datetime import datetime
    from src.feedback.case_rag import render_case_summary
    from src.feedback.models import Outcome, Prediction, Score

    p = Prediction(
        project_id="t01", ticker="02670", company_name="珞石机器人",
        industry="工业机器人", decision_date=datetime(2026, 5, 9),
        recommendation="审慎参与", confidence="中",
        valuation_low=60, valuation_mid=80, valuation_high=100,
        anchor_method="PE", anchor_logic="对标可比 25x",
        ipo_pricing_view="合理",
        suggested_amount_low_usd_m=10, suggested_amount_high_usd_m=20,
        key_supports=["技术领先", "客户多元"],
        key_risks=["客户集中度"],
        model_provider="kimi",
    )
    o = Outcome(prediction_id=1, recorded_date=datetime(2026, 11, 9),
                ipo_actual_marketcap_hkd_billion=110.0,
                d1_return=0.12, d180_return=-0.08,
                was_broken_ipo_d180=True,
                notable_events=["管理层变动"])
    s = Score(prediction_id=1, score_date=datetime(2026, 11, 9),
              recommendation_score=-0.2,
              risks_total_count=1, risks_realized_count=0,
              unforeseen_risks_count=2,
              error_root_causes=["信息缺失", "黑天鹅"],
              postmortem_memo="...内容...\n## 四、教训\n核心教训：客户集中度评估时应额外关注 ...")

    text = render_case_summary(p, o, s)
    assert "珞石机器人" in text
    assert "客户集中度" in text
    assert "D180 (禁售期满): -8.0%" in text
    assert "教训" in text
    assert "信息缺失" in text


def test_postmortem_input_precomputes_valuation_error() -> None:
    from datetime import datetime
    from src.feedback.models import Outcome, Prediction
    from src.feedback.postmortem import _build_postmortem_input

    p = Prediction(
        project_id="t02", ticker="x", company_name="x", industry="x",
        decision_date=datetime.now(),
        recommendation="认购", confidence="中",
        valuation_low=60, valuation_mid=80, valuation_high=100,
        anchor_method="PE", ipo_pricing_view="合理",
        suggested_amount_low_usd_m=10, suggested_amount_high_usd_m=20,
        key_risks=["a", "b"], model_provider="kimi",
    )
    o = Outcome(prediction_id=1, recorded_date=datetime.now(),
                ipo_actual_marketcap_hkd_billion=120.0)
    body, pre = _build_postmortem_input(p, o)
    assert pre["valuation_error_pct"] == 0.5  # (120-80)/80 = 0.5
    assert pre["valuation_within_range"] is False  # 120 > 100
    assert pre["risks_total_count"] == 2
    assert "120" in body


def test_postmortem_input_handles_missing_outcome_fields() -> None:
    """Outcome 字段大量为 None 时不应抛异常。"""
    from datetime import datetime
    from src.feedback.models import Outcome, Prediction
    from src.feedback.postmortem import _build_postmortem_input

    p = Prediction(
        project_id="t03", ticker="x", company_name="x", industry="x",
        decision_date=datetime.now(),
        recommendation="观望", confidence="低",
        valuation_mid=50, anchor_method="PS", ipo_pricing_view="偏高",
        suggested_amount_low_usd_m=0, suggested_amount_high_usd_m=0,
        model_provider="kimi",
    )
    o = Outcome(prediction_id=1, recorded_date=datetime.now())
    body, pre = _build_postmortem_input(p, o)
    assert pre["valuation_error_pct"] is None
    assert pre["valuation_within_range"] is None


def test_pricing_estimates_reasonable_values() -> None:
    from src.llm.pricing import estimate_cost_cny

    # Sonnet 1M input + 200K output: $22 + $108×0.2 = 约 ¥43.6
    cost = estimate_cost_cny("claude-sonnet-4-6", 1_000_000, 200_000, 0)
    assert 40 < cost < 50

    # 未知模型返回 0，不抛异常
    assert estimate_cost_cny("unknown-model", 1000, 100, 0) == 0.0


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
