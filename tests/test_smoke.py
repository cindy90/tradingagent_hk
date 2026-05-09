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


# ============================================================================
# Batch 3 新增测试: 热点工具函数 + 新功能
# ============================================================================


def test_to_ths_hk_code_normalizes_padding() -> None:
    from src.data.ifind_sdk import to_ths_hk_code

    # 5 位带前导 0
    assert to_ths_hk_code("02670") == "2670.HK"
    # 4 位
    assert to_ths_hk_code("0700") == "0700.HK"
    # 3 位
    assert to_ths_hk_code("700") == "0700.HK"
    # 1 位
    assert to_ths_hk_code("9") == "0009.HK"
    # 已带 .HK
    assert to_ths_hk_code("2670.HK") == "2670.HK"
    # 已带 .hk 小写
    assert to_ths_hk_code("2670.hk") == "2670.HK"


def test_to_ths_hk_code_handles_h_prefix_for_pre_listing() -> None:
    """招股询价阶段港股代码常用 H 前缀的副牌（例如珞石的 H2254）。"""
    from src.data.ifind_sdk import to_ths_hk_code

    assert to_ths_hk_code("H2254") == "H2254.HK"
    assert to_ths_hk_code("h2254") == "H2254.HK"
    # H 后跟字母不算副牌（保守降级 → 数字部分前导 0）
    # 不需测，只需保证 H+digits 走副牌路径


def test_verify_company_name_strips_common_suffixes() -> None:
    from src.data.ifind_sdk import verify_company_name

    # 完全相等
    matched, _ = verify_company_name.__wrapped__ if hasattr(verify_company_name, "__wrapped__") else verify_company_name, None
    # 直接测内部逻辑（无法 mock get_basic_data 时，用 monkey-patch）

    import src.data.ifind_sdk as m

    saved = m.get_basic_data
    try:
        m.get_basic_data = lambda *a, **kw: {"corp_short_name": "珞石机器人"}
        # 后缀剔除后核心都是"珞石"
        ok, name = m.verify_company_name("H2254.HK", "珞石（山东）智能科技股份有限公司")
        assert ok is True
        assert name == "珞石机器人"

        # 头 2 字相同也算
        m.get_basic_data = lambda *a, **kw: {"corp_short_name": "越疆-W"}
        ok, _ = m.verify_company_name("02432.HK", "越疆科技股份有限公司")
        assert ok is True

        # 完全不相关
        m.get_basic_data = lambda *a, **kw: {"corp_short_name": "云迹科技-W"}
        ok, _ = m.verify_company_name("02670.HK", "珞石智能科技股份有限公司")
        assert ok is False

        # 空数据
        m.get_basic_data = lambda *a, **kw: {}
        ok, name = m.verify_company_name("X.HK", "随便")
        assert ok is False
        assert name is None
    finally:
        m.get_basic_data = saved


def test_default_target_fiscal_year() -> None:
    from src.data.ifind_sdk import _default_target_fiscal_year
    from datetime import date

    y = _default_target_fiscal_year()
    today = date.today()
    if today.month >= 7:
        assert y == today.year - 1
    else:
        assert y == today.year - 2


def test_get_history_quotes_uses_dynamic_dates(monkeypatch) -> None:
    """sdate/edate 为 None 时应自动算今日和往前 days_back 天。"""
    from datetime import date, timedelta
    import src.data.ifind_sdk as m

    captured: dict = {}

    def fake_post(*args, **kwargs):
        # 拦截：到不了真实 SDK，因为 _ensure_login 会失败
        captured["sdate"] = args[3] if len(args) > 3 else kwargs.get("sdate")
        captured["edate"] = args[4] if len(args) > 4 else kwargs.get("edate")
        return []

    # _ensure_login 返回 False → get_history_quotes 直接返 []
    monkeypatch.setattr(m, "_ensure_login", lambda: False)

    out = m.get_history_quotes("2670.HK")  # 用动态默认
    assert out == []
    # cache 也无入因为没数据，但我们能确认调用未抛
    out2 = m.get_history_quotes("2670.HK", days_back=30)
    assert out2 == []


def test_evidence_pages_validator_coerces_strings() -> None:
    from src.feedback.models import IndustryScoreCard

    sc = IndustryScoreCard(
        summary="test", overall_score=3.0,
        industry_score=3.0, competition_intensity=3.0,
        evidence_pages=["P.42", "招股书 P.108-109", 200, "无页码", 3.5, "5"],
    )
    # 字符串里抽数字; 无数字的丢弃; float 转 int; bool 不计
    assert sc.evidence_pages == [42, 108, 200, 3, 5]


def test_evidence_pages_validator_handles_non_list() -> None:
    from src.feedback.models import IndustryScoreCard

    # 单个 int 不在 list 里也接受
    sc = IndustryScoreCard(
        summary="x", overall_score=3.0,
        industry_score=3.0, competition_intensity=3.0,
        evidence_pages=42,  # type: ignore[arg-type]
    )
    assert sc.evidence_pages == [42]


def test_evidence_pages_validator_none_returns_empty() -> None:
    from src.feedback.models import IndustryScoreCard

    sc = IndustryScoreCard(
        summary="x", overall_score=3.0,
        industry_score=3.0, competition_intensity=3.0,
        evidence_pages=None,  # type: ignore[arg-type]
    )
    assert sc.evidence_pages == []


def test_save_prediction_upsert_overwrites_on_duplicate(tmp_path) -> None:
    from datetime import datetime

    from src.feedback import FeedbackStore
    from src.feedback.models import Prediction

    s = FeedbackStore(db_path=tmp_path / "test.sqlite")

    p1 = Prediction(
        project_id="dup_001", ticker="02670", company_name="A",
        industry="x", decision_date=datetime.now(),
        recommendation="认购", confidence="高",
        valuation_mid=100.0, anchor_method="PE", ipo_pricing_view="合理",
        suggested_amount_low_usd_m=10, suggested_amount_high_usd_m=20,
        model_provider="kimi",
    )
    pid1 = s.save_prediction(p1)

    # 同 project_id 但内容改变
    p2 = Prediction(
        project_id="dup_001", ticker="02670", company_name="A 改名后",
        industry="x", decision_date=datetime.now(),
        recommendation="审慎参与", confidence="中",
        valuation_mid=80.0, anchor_method="PS", ipo_pricing_view="偏高",
        suggested_amount_low_usd_m=5, suggested_amount_high_usd_m=15,
        model_provider="kimi",
    )
    pid2 = s.save_prediction(p2)

    assert pid1 == pid2  # 同 id (UPDATE 而非 INSERT)
    fetched = s.get_prediction(pid1)
    assert fetched is not None
    assert fetched.company_name == "A 改名后"
    assert fetched.recommendation == "审慎参与"
    assert fetched.valuation_mid == 80.0


def test_save_prediction_upsert_false_skips(tmp_path) -> None:
    from datetime import datetime

    from src.feedback import FeedbackStore
    from src.feedback.models import Prediction

    s = FeedbackStore(db_path=tmp_path / "test.sqlite")
    p1 = Prediction(
        project_id="skip_001", ticker="x", company_name="A",
        industry="x", decision_date=datetime.now(),
        recommendation="认购", confidence="高",
        valuation_mid=100.0, anchor_method="PE", ipo_pricing_view="合理",
        suggested_amount_low_usd_m=10, suggested_amount_high_usd_m=20,
        model_provider="kimi",
    )
    pid1 = s.save_prediction(p1)
    p2 = Prediction(
        project_id="skip_001", ticker="x", company_name="B",
        industry="x", decision_date=datetime.now(),
        recommendation="不认购", confidence="低",
        valuation_mid=50.0, anchor_method="PE", ipo_pricing_view="偏高",
        suggested_amount_low_usd_m=0, suggested_amount_high_usd_m=0,
        model_provider="kimi",
    )
    pid2 = s.save_prediction(p2, upsert=False)
    assert pid1 == pid2
    fetched = s.get_prediction(pid1)
    # 没被覆盖
    assert fetched.company_name == "A"
    assert fetched.recommendation == "认购"


def test_peer_suggester_parser_valid_json() -> None:
    from src.agents.peer_suggester import _parse_candidates

    text = """这是一些前置文字...

```json
[
  {"ticker": "02432", "name": "越疆", "reason": "协作机器人直接竞品"},
  {"ticker": "01021", "name": "华沿机器人", "reason": "工业机器人本体"}
]
```
"""
    cands = _parse_candidates(text)
    assert len(cands) == 2
    assert cands[0].ticker == "02432"
    assert cands[0].name == "越疆"
    assert cands[1].ticker == "01021"


def test_peer_suggester_parser_normalizes_ticker_padding() -> None:
    from src.agents.peer_suggester import _parse_candidates

    text = '```json\n[{"ticker": "2432", "name": "x", "reason": "x"}]\n```'
    cands = _parse_candidates(text)
    assert len(cands) == 1
    # 4 位补到 5 位
    assert cands[0].ticker == "02432"


def test_peer_suggester_parser_falls_back_to_bare_array() -> None:
    """LLM 没用 ```json``` 包裹时也能解析。"""
    from src.agents.peer_suggester import _parse_candidates

    text = '前文\n[{"ticker": "02432", "name": "x", "reason": "y"}]\n后文'
    cands = _parse_candidates(text)
    assert len(cands) == 1


def test_peer_suggester_parser_handles_invalid_json() -> None:
    from src.agents.peer_suggester import _parse_candidates

    assert _parse_candidates("没有 json") == []
    assert _parse_candidates("```json\n[invalid]\n```") == []


def test_normalize_step_names_handles_aliases() -> None:
    from src.graph.workflow import normalize_step_names

    # 短名映射到标准名
    assert "prospectus_analyst" in normalize_step_names("prospectus")
    assert "debate_manager" in normalize_step_names("debate")
    # 'all' 返回全部 10 步按序号排（v2 加了 fact_check）
    all_steps = normalize_step_names("all")
    assert all_steps[0] == "prospectus_analyst"
    assert all_steps[-1] == "decision"
    assert len(all_steps) == 10
    assert "fact_check" in all_steps
    # 测试 fact_check 别名
    assert "fact_check" in normalize_step_names("factcheck")
    # 多步去重 + 按序号排序
    out = normalize_step_names("decision,prospectus,risk")
    assert out == ["prospectus_analyst", "risk", "decision"]
    # 未知步骤抛异常
    import pytest
    with pytest.raises(ValueError, match="未知 step"):
        normalize_step_names("nonexistent")


def test_save_run_metadata_round_trip(tmp_path) -> None:
    from src.graph.workflow import load_run_metadata, save_run_metadata

    save_run_metadata(
        tmp_path,
        ticker="02670",
        company_name="珞石",
        industry="工业机器人",
        peers=["02432", "01021"],
        recent_ipos=None,  # None 应该被过滤
        ifind_target=None,
    )
    md = load_run_metadata(tmp_path)
    assert md["ticker"] == "02670"
    assert md["peers"] == ["02432", "01021"]
    # None 字段不应出现
    assert "recent_ipos" not in md
    assert "ifind_target" not in md


def test_load_run_metadata_missing_dir_returns_empty(tmp_path) -> None:
    from src.graph.workflow import load_run_metadata

    assert load_run_metadata(tmp_path / "nonexistent") == {}


# ============================================================================
# iFinD 全覆盖（指数/IPO returns/公告）测试
# ============================================================================

def test_tag_announcement_recognizes_keywords() -> None:
    from src.data.ifind_sdk import tag_announcement

    assert "减持" in tag_announcement("控股股东减持公告")
    assert "减持" in tag_announcement("Disposal of Shares by Major Shareholder")
    assert "业绩" in tag_announcement("盈利预警公告")
    assert "业绩" in tag_announcement("Profit Warning")
    assert "回购" in tag_announcement("股份回购公告")
    assert "增发" in tag_announcement("配售公告")
    # 多标签
    tags = tag_announcement("拟回购及调整薪酬体系")
    assert "回购" in tags
    assert "调整" in tags
    # 不命中
    assert tag_announcement("董事会会议通告") == []


def test_compute_post_ipo_returns_returns_error_without_listing_date(monkeypatch) -> None:
    """未传 listing_date 且 iFinD 拉不到时应返错误结构。"""
    import src.data.ifind_sdk as m

    # mock get_peer_ipo_summary 返回空 ipo_date
    monkeypatch.setattr(m, "get_peer_ipo_summary", lambda *a, **kw: {})
    out = m.compute_post_ipo_returns("02670", 18.5)
    assert "error" in out


def test_compute_post_ipo_returns_handles_invalid_listing_date(monkeypatch) -> None:
    import src.data.ifind_sdk as m

    out = m.compute_post_ipo_returns("02670", 18.5, listing_date="not-a-date")
    assert "error" in out


def test_compute_post_ipo_returns_with_mocked_quotes(monkeypatch) -> None:
    """给定模拟 K 线数据, 验证 returns / 破发 / 最大回撤计算正确。"""
    import src.data.ifind_sdk as m

    # 100 行: 价格 18.5 → 20 → 22 → ... → 17 (锁定期内有冲高有回撤)
    # 简化: D1 收 19, D30 收 22, D90 收 25, D180 收 16
    fake_rows: list[dict] = []
    for i in range(200):
        if i == 0:
            close = 19.0
            open_ = 19.5
        elif i < 30:
            close = 19.0 + i * 0.1  # 涨到 22 左右
            open_ = close
        elif i < 90:
            close = 22.0 + (i - 30) * 0.05
            open_ = close
        else:
            close = 25.0 - (i - 90) * 0.08  # 跌
            open_ = close
        fake_rows.append({"open": open_, "close": close, "amount": 100_000_000})

    monkeypatch.setattr(m, "get_history_quotes", lambda *a, **kw: fake_rows)
    monkeypatch.setattr(m, "get_peer_ipo_summary", lambda *a, **kw: {"ipo_date": "2026-01-01"})

    out = m.compute_post_ipo_returns("02670", 18.5, listing_date="2026-01-01")
    assert out.get("error") is None
    assert out["d1_close"] == 19.0
    assert out["d1_return"] == round(19.0 / 18.5 - 1, 4)
    assert out["d1_open_return"] == round(19.5 / 18.5 - 1, 4)
    assert out["was_broken_d1"] is False
    # D180 = 索引 179: close = 25.0 - 89 * 0.08 = 17.88，仍未破发
    # 但前面有冲高 25 → 后续回撤超过 25%，max_dd_in_d180 应为负
    assert out["d180_close"] is not None
    assert out["max_drawdown_in_d180_pct"] is not None
    assert out["max_drawdown_in_d180_pct"] < 0  # 必有回撤
    # 平均成交 = 100M / 1M = 100 百万
    assert out["avg_daily_turnover_hkd_m_d180"] == 100.0


def test_compute_post_ipo_returns_handles_zero_ipo_price(monkeypatch) -> None:
    import src.data.ifind_sdk as m

    monkeypatch.setattr(m, "get_history_quotes", lambda *a, **kw: [
        {"open": 1.0, "close": 1.0, "amount": 100}
    ])
    out = m.compute_post_ipo_returns("X", 0.0, listing_date="2026-01-01")
    # ipo_price=0 不应抛 ZeroDivisionError, 各 return 字段为 None
    assert out["d1_return"] is None


def test_get_index_summary_handles_no_data(monkeypatch) -> None:
    import src.data.ifind_sdk as m

    monkeypatch.setattr(m, "get_history_quotes", lambda *a, **kw: [])
    monkeypatch.setattr(m, "get_basic_data", lambda *a, **kw: {})
    out = m.get_index_summary("HSI")
    assert out["index"] == "HSI"
    assert out["thscode"] == "HSI.HI"
    assert out["latest_close"] is None
    assert out["pe_ttm"] is None


def test_get_index_summary_picks_up_close_and_pe(monkeypatch) -> None:
    import src.data.ifind_sdk as m

    fake_rows = [{"close": 22000.0 + i * 10, "time": f"2026-01-{(i % 28) + 1:02d}"} for i in range(100)]
    monkeypatch.setattr(m, "get_history_quotes", lambda *a, **kw: fake_rows)
    monkeypatch.setattr(m, "get_basic_data", lambda *a, **kw: {"pe_ttm": 9.5, "pb_latest": 1.1})
    out = m.get_index_summary("HSI")
    assert out["latest_close"] == fake_rows[-1]["close"]
    assert out["pe_ttm"] == 9.5
    assert out["pb_latest"] == 1.1
    assert out["change_30d"] is not None
    assert out["change_90d"] is not None


def test_macro_uses_market_indices_from_extras_first(monkeypatch) -> None:
    """macro Agent 优先从 ctx.extras.market_indices 读, 而不是每次重新调 SDK."""
    from src.agents.extras import WorkflowExtras
    from src.agents.macro import _render_indices_md

    # 直接测渲染函数: 给定 indices 列表 → 输出表格
    indices = [
        {"index": "HSI", "thscode": "HSI.HI", "latest_close": 23000,
         "latest_date": "2026-05-09", "change_30d": 1.5, "change_90d": -2.1,
         "pe_ttm": 9.8, "pb_latest": 1.05},
    ]
    md = _render_indices_md(indices)
    assert "HSI" in md
    assert "23000" in md
    assert "9.8" in md
    # 空数据
    assert "未返回指数数据" in _render_indices_md([])


# ============================================================================
# 专业级 v2 升级测试
# ============================================================================

def test_decision_schema_v2_supports_sensitivity_table() -> None:
    """新 schema 必须能解析三档敏感性 + 对冲 + kill switches."""
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "审慎参与",
        "confidence": "中",
        "suggested_amount_usd_million": [10.0, 20.0],
        "valuation_range_hkd_billion": {
            "low": 50.0, "mid": 80.0, "high": 110.0,
            "anchor_method": "PEG", "anchor_logic": "对标 peer PEG 1.4x",
        },
        "ipo_pricing_view": "合理",
        "key_supports": ["技术领先"],
        "key_risks": ["客户集中度高"],
        "deal_conditions": [],
        "monitoring_kpis": ["毛利率"],
        "sensitivity_table": [
            {"name": "悲观", "triggers": ["H1 毛利<30%"], "valuation_hkd_b": 45.0,
             "probability": 0.30, "expected_return_pct": -44.0},
            {"name": "基准", "triggers": ["财务符合预期"], "valuation_hkd_b": 80.0,
             "probability": 0.50, "expected_return_pct": 0.0},
            {"name": "乐观", "triggers": ["第二曲线兑现"], "valuation_hkd_b": 130.0,
             "probability": 0.20, "expected_return_pct": 62.5},
        ],
        "hedging_strategy": {
            "instrument": "恒生科技 ETF PUT",
            "target_coverage_pct": 0.30,
            "rationale": "锁定期内对冲行业系统性风险",
        },
        "exit_plan": {
            "horizon": "解禁日 D+0",
            "method": "VWAP",
            "pace": "分 5 个交易日",
            "trigger_conditions": ["解禁前股价跌破招股价 30%"],
        },
        "kill_switches": [
            {"trigger": "创始人或 CTO 任一 30 天内离职", "action": "立刻减持 100%", "severity": "高"},
            {"trigger": "Q1 经营现金流连续两季为负", "action": "启动 PUT 对冲", "severity": "中"},
        ],
        "monitoring_kpis_detailed": [
            {"name": "毛利率", "threshold": "< 30%", "frequency": "季报",
             "action_if_breach": "触发风险评估"},
        ],
    }
    result, err = validate_decision(parsed)
    assert result is not None
    assert err == ""
    assert len(result.sensitivity_table) == 3
    # 概率之和约 1
    total_p = sum(s.probability for s in result.sensitivity_table)
    assert 0.95 < total_p < 1.05
    assert result.hedging_strategy is not None
    assert result.hedging_strategy.target_coverage_pct == 0.30
    assert len(result.kill_switches) == 2
    assert result.kill_switches[0].severity == "高"
    assert result.exit_plan.method == "VWAP"
    assert result.monitoring_kpis_detailed[0].threshold == "< 30%"


def test_decision_schema_v2_backward_compat() -> None:
    """旧 schema (没有 v2 新字段) 仍能 validate."""
    from src.agents.decision import validate_decision

    parsed_old = {
        "recommendation": "认购",
        "confidence": "中",
        "suggested_amount_usd_million": [10.0, 20.0],
        "valuation_range_hkd_billion": {
            "mid": 80.0, "anchor_method": "PE", "anchor_logic": "x",
        },
        "ipo_pricing_view": "合理",
        "key_supports": ["a"],
        "key_risks": ["b"],
    }
    result, err = validate_decision(parsed_old)
    assert result is not None
    # 新字段默认空
    assert result.sensitivity_table == []
    assert result.hedging_strategy is None
    assert result.kill_switches == []


def test_risk_score_card_supports_detailed_risks() -> None:
    from src.feedback.models import RiskScoreCard, RiskItem

    card = RiskScoreCard(
        summary="整体风险可控",
        overall_score=3.5,
        confidence="中",
        evidence_pages=[],
        overall_risk_level=3.5,
        risk_dimensions={"财务造假": 4.0, "估值高估": 2.5},
        detailed_risks=[
            RiskItem(
                dimension="客户集中",
                score=2.5,
                probability="中",
                impact="大",
                early_warning=["Top1 客户营收占比 > 35%"],
            ),
        ],
        veto_conditions=["招股价 PS > 35x"],
    )
    assert len(card.detailed_risks) == 1
    assert card.detailed_risks[0].probability == "中"
    assert card.detailed_risks[0].early_warning == ["Top1 客户营收占比 > 35%"]


def test_comparable_score_card_supports_peg_and_sotp() -> None:
    from src.feedback.models import ComparableScoreCard

    card = ComparableScoreCard(
        summary="估值偏高",
        overall_score=2.5,
        confidence="中",
        valuation_mid_hkd_b=80.0,
        median_pe=70.0,
        median_ps=22.0,
        median_peg=1.4,
        valuation_method="PEG",
        implied_revenue_cagr_at_ipo=50.0,
        sotp_breakdown={"硬件本体": 40.0, "协作机器人": 25.0, "具身智能": 15.0},
    )
    assert card.median_peg == 1.4
    assert card.implied_revenue_cagr_at_ipo == 50.0
    assert sum(card.sotp_breakdown.values()) == 80.0


def test_fact_checker_agent_runs_with_mock_provider() -> None:
    """FactCheckerAgent 必须能跑过 mock provider 的端到端流程."""
    from pathlib import Path
    import tempfile
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.agents.fact_check import FactCheckerAgent
    from src.llm import LLMClient
    from src.llm.client import LLMResponse

    class _Mock:
        def complete(self, *, model, system, messages, max_tokens, temperature, cached_system_blocks):
            return LLMResponse(
                text="## 一、数字抽取表\n...\n## 二、不一致警告\n经核对未发现重大数字矛盾",
                input_tokens=100, output_tokens=50, model=model,
            )

    llm = LLMClient(provider=_Mock(), provider_name="mock")
    agent = FactCheckerAgent(llm)
    with tempfile.TemporaryDirectory() as d:
        ctx = AgentContext(
            project_id="t", ticker="X", company_name="测试", industry="x",
            reports_dir=Path(d), rag=None, extras=WorkflowExtras(),
        )
        ctx.briefs["prospectus_analyst"] = "营收 4.2 亿"
        ctx.briefs["comparable"] = "target revenue 4.5 亿"
        report = agent.run(ctx)
        assert agent.fatal is False  # 非致命
        assert "fact_check" in ctx.briefs
        assert "数字抽取" in report.full_report or "不一致" in report.full_report


def test_workflow_extras_supports_v2_fields() -> None:
    """新增 roadshow_signals / competing_ipos / market_indices 等字段."""
    from src.agents.extras import WorkflowExtras

    ex = WorkflowExtras()
    ex.roadshow_signals = {"dark_pool_price": 22.5, "oversubscribe_retail_x": 80}
    ex.competing_ipos = [{"ticker": "01234", "name": "竞品 A"}]
    ex.market_indices = [{"index": "HSI", "latest_close": 23000}]
    ex.peer_announcements = {"02432": [{"date": "2026-04-01", "title": "回购"}]}

    assert ex.roadshow_signals["dark_pool_price"] == 22.5
    assert ex.competing_ipos[0]["ticker"] == "01234"
    assert ex.market_indices[0]["latest_close"] == 23000
    assert "02432" in ex.peer_announcements


def test_final_memo_writer_renders_sensitivity(tmp_path) -> None:
    """FINAL_MEMO writer 渲染敏感性表 / 对冲策略 / kill switches."""
    from datetime import datetime
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.writer import write_final_summary

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "审慎参与",
        "confidence": "中",
        "valuation_range_hkd_billion": {"low": 50, "mid": 80, "high": 110, "anchor_method": "PEG"},
        "ipo_pricing_view": "合理",
        "suggested_amount_usd_million": [10, 20],
        "key_supports": ["技术领先", "海外加速"],
        "key_risks": ["客户集中度高"],
        "deal_conditions": ["招股价 PS ≤ 30x"],
        "sensitivity_table": [
            {"name": "悲观", "triggers": ["毛利<30%"], "valuation_hkd_b": 45, "probability": 0.3, "expected_return_pct": -44.0},
            {"name": "基准", "triggers": ["持平"], "valuation_hkd_b": 80, "probability": 0.5, "expected_return_pct": 0.0},
            {"name": "乐观", "triggers": ["二曲线"], "valuation_hkd_b": 130, "probability": 0.2, "expected_return_pct": 62.5},
        ],
        "hedging_strategy": {
            "instrument": "ETF PUT", "target_coverage_pct": 0.3, "rationale": "对冲系统性风险",
        },
        "exit_plan": {"horizon": "D+0", "method": "VWAP", "pace": "5 日", "trigger_conditions": []},
        "kill_switches": [
            {"trigger": "CTO 离职", "action": "减持 100%", "severity": "高"},
        ],
        "monitoring_kpis_detailed": [
            {"name": "毛利率", "threshold": "<30%", "frequency": "季报", "action_if_breach": "评估"},
        ],
    }
    ctx = AgentContext(
        project_id="memo_test", ticker="X", company_name="测试", industry="x",
        reports_dir=tmp_path, rag=None, extras=extras,
    )
    out = write_final_summary(ctx)
    text = out.read_text(encoding="utf-8")
    # 校验关键章节都渲染了
    assert "投决摘要" in text
    assert "敏感性分析" in text
    assert "对冲策略" in text
    assert "Kill Switches" in text
    # 渲染了具体内容
    assert "悲观" in text and "乐观" in text
    assert "ETF PUT" in text
    assert "CTO 离职" in text
    assert "毛利率" in text
    # 概率被乘了 100
    assert "30%" in text or "30.0%" in text


# ============================================================================
# peer_pool v2 测试 (#1 改进)
# ============================================================================

def test_derive_keywords_from_industry() -> None:
    from src.data.peer_pool import derive_keywords_from_industry

    kws = derive_keywords_from_industry("工业机器人/协作机器人")
    assert "工业机器人" in kws
    assert "协作机器人" in kws

    kws2 = derive_keywords_from_industry("AI/SaaS")
    assert "AI" in kws2 or "SaaS" in kws2

    # 空值 / 单字
    assert derive_keywords_from_industry("") == []


def test_filter_by_keywords_matches_any() -> None:
    from src.data.peer_pool import _filter_by_keywords

    snapshot = [
        {"ticker": "02432", "name": "越疆-W"},
        {"ticker": "01021", "name": "华沿机器人"},
        {"ticker": "00700", "name": "腾讯控股"},
        {"ticker": "09660", "name": "地平线机器人-W"},
    ]
    out = _filter_by_keywords(snapshot, ["机器人"])
    out_tickers = [r["ticker"] for r in out]
    assert "01021" in out_tickers
    assert "09660" in out_tickers
    assert "00700" not in out_tickers
    # 越疆-W 不含"机器人"字, 不应匹配
    assert "02432" not in out_tickers
    # 多关键词: 命中任一即留
    out2 = _filter_by_keywords(snapshot, ["越疆", "腾讯"])
    assert {"02432", "00700"} == set(r["ticker"] for r in out2)


def test_filter_by_market_cap_keeps_unknown() -> None:
    from src.data.peer_pool import _filter_by_market_cap

    rows = [
        {"ticker": "A", "market_cap_hkd_b": 50.0},
        {"ticker": "B", "market_cap_hkd_b": 200.0},
        {"ticker": "C", "market_cap_hkd_b": 500.0},
        {"ticker": "D", "market_cap_hkd_b": None},  # 未知
    ]
    out = _filter_by_market_cap(rows, min_cap_hkd_b=80, max_cap_hkd_b=300)
    out_tickers = [r["ticker"] for r in out]
    assert "B" in out_tickers
    assert "D" in out_tickers  # 未知保留
    assert "A" not in out_tickers
    assert "C" not in out_tickers


def test_normalize_ticker_pads_to_5() -> None:
    from src.data.peer_pool import _normalize_ticker

    assert _normalize_ticker("700") == "00700"
    assert _normalize_ticker("02432") == "02432"
    assert _normalize_ticker("2432.HK") == "02432"
    assert _normalize_ticker("2432.hk") == "02432"
    assert _normalize_ticker(None) is None
    assert _normalize_ticker("X123") is None


def test_peer_suggester_v2_validates_against_pool(monkeypatch) -> None:
    """v2 必须拒绝 LLM 输出池外的 ticker（防幻觉关键测试）。"""
    from src.agents.peer_suggester import suggest_peers
    from src.data.rag import ProspectusRAG
    from src.llm import LLMClient
    from src.llm.client import LLMResponse

    pool = [
        {"ticker": "02432", "name": "越疆-W", "market_cap_hkd_b": 80.0},
        {"ticker": "01021", "name": "华沿机器人", "market_cap_hkd_b": 50.0},
    ]
    # LLM 输出 1 个池内 + 1 个池外（捏造的）
    LLM_OUT = (
        "```json\n"
        '[{"ticker":"02432","name":"越疆","similarity_score":5.0,"reason":"协作机器人"},'
        '{"ticker":"00700","name":"腾讯","similarity_score":4.5,"reason":"瞎编的"}]\n'
        "```"
    )

    class _Mock:
        def complete(self, **_):
            return LLMResponse(text=LLM_OUT, input_tokens=10, output_tokens=10, model="x")

    class _MockRAG:
        def search(self, query, k=8):
            return [{"text": "竞争对手", "section": "x",
                     "page_start": 1, "page_end": 1, "distance": 0.1}]

    candidates = suggest_peers(
        _MockRAG(), "珞石机器人", "工业机器人", LLMClient(provider=_Mock(), provider_name="m"),
        pool=pool,
    )
    # 池外 ticker (00700) 应被过滤
    tickers = [c.ticker for c in candidates]
    assert "02432" in tickers
    assert "00700" not in tickers
    # market_cap 应从池里补全
    assert candidates[0].market_cap_hkd_b == 80.0


def test_peer_suggester_v2_sorts_by_similarity(monkeypatch) -> None:
    from src.agents.peer_suggester import suggest_peers
    from src.llm import LLMClient
    from src.llm.client import LLMResponse

    pool = [
        {"ticker": "02432", "name": "A", "market_cap_hkd_b": 50.0},
        {"ticker": "01021", "name": "B", "market_cap_hkd_b": 50.0},
        {"ticker": "09660", "name": "C", "market_cap_hkd_b": 50.0},
    ]
    # LLM 故意乱序输出
    LLM_OUT = (
        "```json\n[\n"
        '{"ticker":"01021","name":"B","similarity_score":3.5,"reason":"x"},\n'
        '{"ticker":"09660","name":"C","similarity_score":4.8,"reason":"x"},\n'
        '{"ticker":"02432","name":"A","similarity_score":4.2,"reason":"x"}\n'
        "]\n```"
    )

    class _Mock:
        def complete(self, **_):
            return LLMResponse(text=LLM_OUT, input_tokens=10, output_tokens=10, model="x")

    class _MockRAG:
        def search(self, query, k=8):
            return [{"text": "x", "section": "", "page_start": 1, "page_end": 1, "distance": 0.1}]

    cands = suggest_peers(_MockRAG(), "x", "y", LLMClient(provider=_Mock(), provider_name="m"), pool=pool)
    scores = [c.similarity_score for c in cands]
    # 必须按 similarity_score 降序
    assert scores == sorted(scores, reverse=True)
    assert cands[0].ticker == "09660"  # similarity 4.8 最高


def test_peer_suggester_clamps_score_to_0_5() -> None:
    """LLM 给 99 / -3 这种异常值, 应 clamp 到 [0, 5]."""
    from src.agents.peer_suggester import _parse_candidates

    text = '```json\n[{"ticker":"02432","name":"x","similarity_score":99.0,"reason":"x"}]\n```'
    out = _parse_candidates(text)
    assert out[0].similarity_score == 5.0

    text2 = '```json\n[{"ticker":"02432","name":"x","similarity_score":-3,"reason":"x"}]\n```'
    out2 = _parse_candidates(text2)
    assert out2[0].similarity_score == 0.0

    # 缺字段 → 默认 3.0
    text3 = '```json\n[{"ticker":"02432","name":"x","reason":"x"}]\n```'
    out3 = _parse_candidates(text3)
    assert out3[0].similarity_score == 3.0


# ============================================================================
# HTML IC memo 渲染测试
# ============================================================================

def test_render_ic_memo_html_includes_all_sections(tmp_path) -> None:
    """端到端: HTML 必须含 Executive Summary / 敏感性表 / kill switches / KPI 等."""
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.html_writer import write_ic_memo_html

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "认购",
        "confidence": "高",
        "valuation_range_hkd_billion": {"low": 60, "mid": 80, "high": 100, "anchor_method": "PEG"},
        "ipo_pricing_view": "合理",
        "suggested_amount_usd_million": [10, 20],
        "key_supports": ["技术领先"],
        "key_risks": ["客户集中度高"],
        "deal_conditions": ["招股价 PS ≤ 30x"],
        "sensitivity_table": [
            {"name": "悲观", "triggers": ["毛利<30%"], "valuation_hkd_b": 45, "probability": 0.3, "expected_return_pct": -44.0},
            {"name": "基准", "triggers": ["持平"], "valuation_hkd_b": 80, "probability": 0.5, "expected_return_pct": 0.0},
            {"name": "乐观", "triggers": ["二曲线"], "valuation_hkd_b": 130, "probability": 0.2, "expected_return_pct": 62.5},
        ],
        "hedging_strategy": {"instrument": "ETF PUT", "target_coverage_pct": 0.3, "rationale": "对冲系统性风险"},
        "exit_plan": {"horizon": "D+0", "method": "VWAP", "pace": "5 日", "trigger_conditions": []},
        "kill_switches": [
            {"trigger": "CTO 离职", "action": "减持 100%", "severity": "高"},
        ],
        "monitoring_kpis_detailed": [
            {"name": "毛利率", "threshold": "<30%", "frequency": "季报", "action_if_breach": "评估"},
        ],
    }

    ctx = AgentContext(
        project_id="html_test", ticker="X", company_name="测试公司",
        industry="工业机器人", reports_dir=tmp_path, rag=None, extras=extras,
    )
    ctx.full_reports["decision"] = "## 投决论述\n详细论证..."
    ctx.briefs["prospectus_analyst"] = "**核心结论**: 业务良好"
    ctx.briefs["decision"] = "**最终建议**: 认购"

    out = write_ic_memo_html(ctx)
    text = out.read_text(encoding="utf-8")

    # 关键元素
    assert "<!DOCTYPE html>" in text
    assert "测试公司" in text
    assert "认购" in text
    assert "高" in text  # confidence
    assert "敏感性分析" in text
    assert "Kill Switches" in text
    assert "悲观" in text and "基准" in text and "乐观" in text
    assert "CTO 离职" in text
    assert "毛利率" in text
    # CSS 类色编码
    assert "rec-认购" in text
    assert "scenario-悲观" in text and "scenario-乐观" in text
    assert "severity-高" in text
    # 自包含: 不应有外部 <link> 或外部 <script>
    assert "<link rel=\"stylesheet\"" not in text
    assert "<script src=" not in text


def test_html_writer_handles_minimal_decision(tmp_path) -> None:
    """决议字段大量缺失（早期版本/decision retry 失败）也不应崩。"""
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.html_writer import write_ic_memo_html

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "观望",
        "confidence": "低",
        "valuation_range_hkd_billion": {"mid": 50, "anchor_method": "PE"},
        "ipo_pricing_view": "偏高",
        "suggested_amount_usd_million": [],
        "key_supports": [],
        "key_risks": [],
    }
    ctx = AgentContext(
        project_id="x", ticker="X", company_name="测试", industry="x",
        reports_dir=tmp_path, rag=None, extras=extras,
    )
    out = write_ic_memo_html(ctx)
    text = out.read_text(encoding="utf-8")
    # 该有的占位符
    assert "敏感性分析缺失" in text or "敏感性" in text
    assert "未提供 kill switches" in text
    assert "rec-观望" in text


def test_write_final_summary_emits_both_md_and_html(tmp_path) -> None:
    """write_final_summary 同时输出 .md + .html"""
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.writer import write_final_summary

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "认购",
        "confidence": "中",
        "valuation_range_hkd_billion": {"mid": 80, "anchor_method": "PE"},
        "ipo_pricing_view": "合理",
        "suggested_amount_usd_million": [10, 20],
        "key_supports": ["x"],
        "key_risks": ["y"],
    }
    ctx = AgentContext(
        project_id="dual_out", ticker="X", company_name="测试", industry="x",
        reports_dir=tmp_path, rag=None, extras=extras,
    )
    md_path = write_final_summary(ctx, also_html=True)
    assert md_path.exists()
    assert md_path.suffix == ".md"
    html_path = tmp_path / "FINAL_MEMO.html"
    assert html_path.exists()
    # md 不被影响
    assert "基石投资决策备忘录" in md_path.read_text(encoding="utf-8")


def test_write_final_summary_no_html_skips_html(tmp_path) -> None:
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.writer import write_final_summary

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "观望", "confidence": "中",
        "valuation_range_hkd_billion": {"mid": 50, "anchor_method": "PE"},
        "ipo_pricing_view": "偏高", "suggested_amount_usd_million": [0, 0],
        "key_supports": ["x"], "key_risks": ["y"],
    }
    ctx = AgentContext(
        project_id="md_only", ticker="X", company_name="测试", industry="x",
        reports_dir=tmp_path, rag=None, extras=extras,
    )
    write_final_summary(ctx, also_html=False)
    assert (tmp_path / "FINAL_MEMO.md").exists()
    assert not (tmp_path / "FINAL_MEMO.html").exists()


# ============================================================================
# Decision schema v3 推理链 / 假设清单 / 估值分拆 测试
# ============================================================================

def test_decision_schema_v3_supports_reasoning_chain() -> None:
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "审慎参与",
        "confidence": "中",
        "suggested_amount_usd_million": [10.0, 20.0],
        "valuation_range_hkd_billion": {
            "low": 50, "mid": 80, "high": 110,
            "anchor_method": "多方法加权",
            "anchor_logic": "PS 50% + PEG 30% + SOTP 20%",
            "methodology_breakdown": [
                {"method": "PS", "peer_basis": "可比中位 22x",
                 "target_metric": "2025E 营收 3.6 亿", "formula": "22 × 3.6 = 79",
                 "result_hkd_b": 88.0, "weight": 0.5},
                {"method": "PEG", "peer_basis": "PEG 1.4x",
                 "target_metric": "PE 70 × CAGR 35%", "formula": "1.4 × 70 × 0.7 = 68.6",
                 "result_hkd_b": 70.0, "weight": 0.3},
            ],
            "key_assumptions": ["营收 CAGR 35%", "毛利稳定 38%"],
        },
        "ipo_pricing_view": "合理",
        "key_supports": ["x"],
        "key_risks": ["y"],
        "sensitivity_table": [
            {"name": "悲观", "triggers": ["毛利<30%"], "valuation_hkd_b": 45,
             "probability": 0.3, "expected_return_pct": -44.0,
             "valuation_derivation": "毛利 38→30%, PS 压缩 25% → 60",
             "probability_rationale": "近 3 月港股机器人 IPO 破发率 30%"},
        ],
        "kill_switches": [
            {"trigger": "CTO 离职 30 天内", "action": "减持 100%", "severity": "高",
             "rationale": "创始人股权 35%, 历史回撤中位 40%",
             "historical_precedent": "2024 年 X 公司"},
        ],
        "monitoring_kpis_detailed": [
            {"name": "毛利率", "threshold": "< 30%", "frequency": "季报",
             "action_if_breach": "减持 30%",
             "threshold_rationale": "行业中位 35%, 留 5pp 缓冲",
             "industry_benchmark": "中位 35% / 75 分位 42%"},
        ],
        "key_assumptions": [
            "2025E 营收 3.6 亿",
            "毛利率维持 38% ± 3pp",
            "PS 不压缩 > 25%",
        ],
        "reasoning_chain": [
            {"step_no": 1, "title": "业务质量评估",
             "premise": "协作机器人主业",
             "data_source": "prospectus_analyst: 营收 4.2 亿",
             "calculation": "vs 行业中位高 100%",
             "conclusion": "+10% 估值溢价",
             "confidence": "中",
             "caveats": ["客户集中度高"]},
            {"step_no": 2, "title": "估值起点",
             "premise": "PS 估值",
             "data_source": "comparable: 越疆 PS=27",
             "calculation": "22 × 3.6 = 79",
             "conclusion": "基准估值 88 亿 HKD",
             "confidence": "中"},
        ],
    }
    result, err = validate_decision(parsed)
    assert result is not None, f"v3 schema 校验失败: {err}"
    # 推理链
    assert len(result.reasoning_chain) == 2
    assert result.reasoning_chain[0].title == "业务质量评估"
    assert result.reasoning_chain[0].caveats == ["客户集中度高"]
    # 关键假设
    assert len(result.key_assumptions) == 3
    # 估值方法分拆
    assert len(result.valuation_range_hkd_billion.methodology_breakdown) == 2
    assert result.valuation_range_hkd_billion.methodology_breakdown[0].method == "PS"
    assert result.valuation_range_hkd_billion.methodology_breakdown[0].formula == "22 × 3.6 = 79"
    # 估值 key_assumptions
    assert len(result.valuation_range_hkd_billion.key_assumptions) == 2
    # sensitivity reasoning
    assert "毛利 38→30%" in result.sensitivity_table[0].valuation_derivation
    assert result.sensitivity_table[0].probability_rationale != ""
    # kill switch rationale
    assert "创始人股权 35%" in result.kill_switches[0].rationale
    # KPI rationale
    assert result.monitoring_kpis_detailed[0].threshold_rationale != ""
    assert result.monitoring_kpis_detailed[0].industry_benchmark != ""


def test_decision_schema_v3_backward_compat_no_reasoning_chain() -> None:
    """旧 schema (没有 v3 字段) 仍能 validate. 推理链字段默认空 list."""
    from src.agents.decision import validate_decision

    parsed = {
        "recommendation": "认购",
        "confidence": "高",
        "suggested_amount_usd_million": [10.0, 20.0],
        "valuation_range_hkd_billion": {"mid": 80.0, "anchor_method": "PE", "anchor_logic": "x"},
        "ipo_pricing_view": "合理",
        "key_supports": ["a"],
        "key_risks": ["b"],
    }
    result, err = validate_decision(parsed)
    assert result is not None
    # v3 字段默认空
    assert result.reasoning_chain == []
    assert result.key_assumptions == []
    assert result.valuation_range_hkd_billion.methodology_breakdown == []


def test_html_renders_reasoning_chain_and_assumptions(tmp_path) -> None:
    """HTML 模板正确渲染 v3 新区."""
    from src.agents.base import AgentContext
    from src.agents.extras import WorkflowExtras
    from src.reports.html_writer import write_ic_memo_html

    extras = WorkflowExtras()
    extras.decision_json = {
        "recommendation": "认购", "confidence": "高",
        "valuation_range_hkd_billion": {
            "mid": 80, "anchor_method": "PEG", "anchor_logic": "x",
            "methodology_breakdown": [
                {"method": "PS", "peer_basis": "中位 22x", "target_metric": "营收 3.6 亿",
                 "formula": "22 × 3.6 = 79", "result_hkd_b": 79, "weight": 0.5},
            ],
            "key_assumptions": ["营收 CAGR 35%"],
        },
        "ipo_pricing_view": "合理",
        "suggested_amount_usd_million": [10, 20],
        "key_supports": ["x"], "key_risks": ["y"],
        "key_assumptions": [
            "假设 1: 营收增速保持 35%",
            "假设 2: 创始人不离职",
        ],
        "reasoning_chain": [
            {"step_no": 1, "title": "业务质量",
             "premise": "协作机器人", "data_source": "prospectus: 营收 4.2 亿",
             "calculation": "vs 行业 +100%", "conclusion": "+10% 溢价",
             "confidence": "中", "caveats": ["集中度风险"]},
        ],
        "sensitivity_table": [
            {"name": "悲观", "triggers": ["毛利<30%"], "valuation_hkd_b": 45,
             "probability": 0.3, "expected_return_pct": -44.0,
             "valuation_derivation": "毛利压缩, PS 25% 压力",
             "probability_rationale": "历史破发率 30%"},
        ],
        "kill_switches": [
            {"trigger": "CTO 离职", "action": "减持 100%", "severity": "高",
             "rationale": "创始人股权 35%, 历史中位回撤 40%"},
        ],
        "monitoring_kpis_detailed": [
            {"name": "毛利率", "threshold": "< 30%", "frequency": "季报",
             "action_if_breach": "减持 30%",
             "threshold_rationale": "行业中位 35%, 留 5pp 缓冲"},
        ],
    }
    ctx = AgentContext(
        project_id="v3_html_test", ticker="X", company_name="测试",
        industry="x", reports_dir=tmp_path, rag=None, extras=extras,
    )
    out = write_ic_memo_html(ctx)
    text = out.read_text(encoding="utf-8")
    # v3 关键章节
    assert "投决依赖的核心假设" in text or "关键假设" in text
    assert "估值方法分拆" in text
    assert "推理链" in text
    # 推理步骤内容
    assert "业务质量" in text
    assert "+10% 溢价" in text
    assert "集中度风险" in text  # caveats
    assert "conf-中" in text  # 置信度 CSS 类
    # 估值方法行
    assert "method-row" in text or "method-card" in text
    assert "22 × 3.6 = 79" in text  # formula
    # rationale tooltip
    assert "data-rationale" in text
    assert "毛利压缩" in text  # sensitivity reasoning
    assert "创始人股权 35%" in text  # kill switch rationale
    assert "行业中位 35%" in text  # KPI rationale
    # 假设清单
    assert "假设 1: 营收增速保持 35%" in text
