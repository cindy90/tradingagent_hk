"""基石投资分析主工作流。

执行顺序（每步落盘 markdown 报告，下游 Agent 只读 brief 摘要）：
  1) 招股书深度分析     → prospectus_analyst
  2) 行业研究           → industry
  3) 宏观策略           → macro
  4) 可比公司估值       → comparable
  5) 技术发展趋势       → tech_trend
  6) 二级市场情绪       → sentiment
  7) Bull/Bear 辩论     → debate_manager
  8) 风控独立评估       → risk
  9) 最终投决           → decision

未来可改造为 LangGraph，当前用线性 + 显式步骤，便于调试和理解。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import get_settings
from src.agents.base import AgentContext, BaseAgent
from src.agents.comparable import ComparableAgent
from src.agents.debate import DebateOrchestrator
from src.agents.decision import DecisionAgent
from src.agents.industry import IndustryAgent
from src.agents.macro import MacroAgent
from src.agents.prospectus_analyst import ProspectusAnalystAgent
from src.agents.risk import RiskAgent
from src.agents.sentiment import SentimentAgent
from src.agents.summarizer import Summarizer
from src.agents.tech_trend import TechTrendAgent
from src.agents.extras import WorkflowExtras
from src.data.prospectus import ProspectusChunk, ProspectusLoader
from src.data.rag import ProspectusRAG
from src.data.ths_client import THSClient
from src.llm import LLMClient

# 高信息密度章节关键词（按优先级排序）。命中后作为 cached_system_blocks 内容，
# 跨 Agent 复用 prompt cache，且每个 Agent 都白拿这些核心上下文。
KEY_CACHED_SECTIONS = [
    "概要",       # 招股书第一章，浓缩全文
    "风险因素",
    "业务",
    "募集资金用途",
    "募资",
    "财务资料",
    "管理层讨论",
    "RISK FACTORS",
    "BUSINESS",
    "USE OF PROCEEDS",
]


def select_cached_blocks(
    chunks: list[ProspectusChunk],
    *,
    max_blocks: int = 4,
    max_chars_per_block: int = 6000,
    max_total_chars: int = 20000,
) -> list[str]:
    """从切块中挑选关键章节文本作为 prompt cache 内容。

    匹配规则:
      1) chunk.section 包含 KEY_CACHED_SECTIONS 关键词
      2) 同一 section 只取第一块（避免重复）
      3) 总字符数 < max_total_chars
      4) 没有命中关键词时退化为前 3 块（保留原行为）
    """
    selected: list[str] = []
    seen_sections: set[str] = set()
    total = 0

    for kw in KEY_CACHED_SECTIONS:
        for c in chunks:
            sec = (c.section or "").strip()
            if not sec or sec in seen_sections:
                continue
            if kw not in sec:
                continue
            text = c.text[:max_chars_per_block]
            if total + len(text) > max_total_chars:
                break
            selected.append(text)
            seen_sections.add(sec)
            total += len(text)
            break  # 每个关键词只取一块
        if len(selected) >= max_blocks or total >= max_total_chars:
            break

    if not selected:
        # fallback: 前 3 块（保留旧行为，避免完全没 cache）
        selected = [c.text[:max_chars_per_block] for c in chunks[:3]]

    return selected


def _enrich_peer_with_valuation(peer: dict, year: int = 2025) -> None:
    """给 peer dict 补 PE/PS/PB/EV-EBITDA/EV-Sales + 营收/净利/毛利率等。原地修改。"""
    from src.data.ifind_sdk import get_peer_valuation_multiples, get_peer_fundamentals

    ticker = peer["ticker"]
    try:
        peer.update({
            k: v for k, v in get_peer_valuation_multiples(ticker).items()
            if k not in ("ticker", "thscode")
        })
    except Exception as e:
        logger.warning(f"[Prefetch SDK] {ticker} 估值倍数失败: {e}")
    try:
        peer.update({
            k: v for k, v in get_peer_fundamentals(ticker, year=year).items()
            if k not in ("ticker", "thscode")
        })
    except Exception as e:
        logger.warning(f"[Prefetch SDK] {ticker} 基本面失败: {e}")


def _prefetch_ifind_sdk(
    peers: list[str],
    recent_ipos: list[str] | None = None,
    target_ticker: str | None = None,
    expected_target_name: str | None = None,
) -> dict[str, Any]:
    """用 iFinD SDK 拉可比公司近期 K 线 + IPO 信息（含首日开盘价）。

    Args:
        peers: 估值可比清单（业务直接相似, 2-4 家）。
        recent_ipos: 近期同行业 IPO 队列（5-10 家, 反映打新情绪）。
            未给时默认 = peers（让 sentiment 至少有一份"近期 IPO 首日表现"数据）。

    返回:
      {
        "peers": [{ticker, thscode, name, ipo_date, ipo_price, ipo_max_price,
                   first_day_open, first_day_open_return}, ...],
        "peer_recent_quotes": [{ticker, thscode, latest_close, latest_date,
                                 return_30d, return_90d}, ...],
        "recent_hk_ipos": [...同 peers 结构, 但范围可能更广...]
      }
    SDK 不可用 / 登录失败时返回空 list 字段。
    """
    from datetime import date, timedelta

    from src.data.ifind_sdk import compute_peer_quote_summary, get_peer_ipo_summary

    today = date.today()
    sdate = (today - timedelta(days=140)).strftime("%Y-%m-%d")  # 多取日历日 → 保 90 交易日
    edate = today.strftime("%Y-%m-%d")

    def _ipo_for(ticker: str) -> dict | None:
        try:
            return get_peer_ipo_summary(ticker)
        except Exception as e:
            logger.warning(f"[Prefetch SDK] {ticker} IPO 信息失败: {e}")
            return None

    # peers: 估值可比 → 同时拉 K 线 + 估值倍数 + 财务基本面
    peers_info: list[dict] = []
    quotes: list[dict] = []
    for ticker in peers:
        ipo = _ipo_for(ticker)
        if ipo is None:
            continue
        try:
            quote = compute_peer_quote_summary(ticker, sdate, edate)
        except Exception as e:
            logger.warning(f"[Prefetch SDK] {ticker} K 线失败: {e}")
            quote = {"ticker": ticker, "thscode": ipo.get("thscode")}
        _enrich_peer_with_valuation(ipo)
        peers_info.append(ipo)
        quotes.append(quote)
        logger.info(
            f"[Prefetch SDK] peer {ticker} ({ipo.get('name')}): "
            f"close={quote.get('latest_close')} PS={ipo.get('ps_ttm')} "
            f"PB={ipo.get('pb_latest')} 营收={ipo.get('revenue')} "
            f"first_day_open_return={ipo.get('first_day_open_return')}%"
        )

    # recent_hk_ipos: 默认 = peers, 用户提供时合并(按 ticker 去重)
    ipo_cohort_tickers = list(peers)
    if recent_ipos:
        for t in recent_ipos:
            if t not in ipo_cohort_tickers:
                ipo_cohort_tickers.append(t)

    if recent_ipos:
        # 仅为新增的拉 IPO summary, peers 部分复用上面的
        existing_by_ticker = {p["ticker"]: p for p in peers_info}
        recent_ipos_info: list[dict] = []
        for ticker in ipo_cohort_tickers:
            if ticker in existing_by_ticker:
                recent_ipos_info.append(existing_by_ticker[ticker])
            else:
                ipo = _ipo_for(ticker)
                if ipo is None:
                    continue
                recent_ipos_info.append(ipo)
                logger.info(
                    f"[Prefetch SDK] recent_ipo {ticker} ({ipo.get('name')}): "
                    f"ipo_date={ipo.get('ipo_date')} ipo_price={ipo.get('ipo_price')} "
                    f"first_day_open={ipo.get('first_day_open')}"
                )
    else:
        # 未额外提供, recent_hk_ipos = peers (引用同一组数据)
        recent_ipos_info = peers_info

    # 同时拉 target 公司（招股股票本身）的估值倍数 + 财务
    # 注意: 招股阶段的港股 ticker 在 iFinD 可能被复用（例如 2670.HK 实际是云迹）。
    # caller 应传 ifind 真实代码（如 H2254 副牌），并配合 expected_target_name 做校验。
    target_data: dict | None = None
    if target_ticker:
        try:
            from src.data.ifind_sdk import to_ths_hk_code, verify_company_name
            thscode = to_ths_hk_code(target_ticker)
            if expected_target_name:
                matched, actual_name = verify_company_name(thscode, expected_target_name)
                if not matched:
                    logger.error(
                        f"[Prefetch SDK] ⚠️ target ticker {target_ticker} 在 iFinD 实际是"
                        f"'{actual_name}', 与 '{expected_target_name}' 不匹配; "
                        f"target 数据可能被错误公司污染, 已跳过 target_valuation。"
                        f"如该项目在 IPO 询价阶段, 请用 --ifind-target 显式指定副牌代码 (如 H2254)。"
                    )
                    target_data = None
                    raise RuntimeError("ticker name mismatch")
                logger.info(f"[Prefetch SDK] target {thscode} 公司名校验通过: {actual_name}")
            target_data = {"ticker": target_ticker, "thscode": thscode}
            _enrich_peer_with_valuation(target_data)
            logger.info(
                f"[Prefetch SDK] target {thscode}: PS={target_data.get('ps_ttm')} "
                f"PB={target_data.get('pb_latest')} 营收={target_data.get('revenue')} "
                f"净利={target_data.get('net_profit')}"
            )
        except Exception as e:
            logger.warning(f"[Prefetch SDK] target {target_ticker} 估值/财务失败: {e}")
            target_data = None

    # peer 近期公告（最近 180 天）— 减持 / 业绩预警 / 回购等
    # 用于 sentiment / risk: 看 peer 是否有"实际控制人减持""盈利预警"等
    # 会影响赛道情绪的事件
    peer_announcements: dict[str, list[dict]] = {}
    try:
        from src.data.ifind_sdk import get_recent_announcements, tag_announcement
        for ticker in peers:
            try:
                anns = get_recent_announcements(ticker, days=180, limit=15)
                for a in anns:
                    a["tags"] = tag_announcement(a.get("title", ""))
                peer_announcements[ticker] = anns
                if anns:
                    n_tagged = sum(1 for a in anns if a["tags"])
                    logger.info(
                        f"[Prefetch] {ticker} 近 180 天公告 {len(anns)} 条 "
                        f"({n_tagged} 条命中关键事件)"
                    )
            except Exception as e:
                logger.warning(f"[Prefetch] {ticker} 公告查询失败: {e}")
    except ImportError:
        pass

    # 港股三大指数（HSI / HSCEI / HSTECH 实时 + PE）→ macro Agent 用
    market_indices: list[dict] = []
    try:
        from src.data.ifind_sdk import get_indices_summary
        market_indices = get_indices_summary(["HSI", "HSCEI", "HSTECH"])
        logger.info(
            f"[Prefetch] 港股指数: "
            + " | ".join(
                f"{r.get('index')}={r.get('latest_close')}"
                for r in market_indices
            )
        )
    except Exception as e:
        logger.warning(f"[Prefetch] 港股指数查询失败: {e}")

    return {
        "peers": peers_info,
        "peer_recent_quotes": quotes,
        "recent_hk_ipos": recent_ipos_info,
        "target_valuation": target_data,
        "peer_announcements": peer_announcements,
        "market_indices": market_indices,
    }


_METADATA_FILENAME = "_run_metadata.json"


def save_run_metadata(reports_dir: Path, **kwargs: Any) -> None:
    """把项目元数据写到 reports_dir/_run_metadata.json，供 rerun 复用。"""
    import json
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / _METADATA_FILENAME
    data = {k: v for k, v in kwargs.items() if v is not None}
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"_run_metadata.json 写入失败: {e}")


def load_run_metadata(reports_dir: Path) -> dict[str, Any]:
    import json
    path = reports_dir / _METADATA_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"_run_metadata.json 读取失败: {e}")
        return {}


# Agent 名 → (实例化函数, 步骤序号, 是否需要 RAG, 是否需要 SDK peers 数据)
_AGENT_REGISTRY: dict[str, tuple[int, bool, bool]] = {
    "prospectus_analyst": (1, True, False),
    "industry": (2, True, True),
    "macro": (3, False, True),
    "comparable": (4, True, True),
    "tech_trend": (5, True, False),
    "sentiment": (6, False, True),
    "fact_check": (7, False, False),  # 新增 #12 数字核对
    "debate_manager": (8, False, False),
    "risk": (9, False, False),
    "decision": (10, False, False),
}

# 用户输入步骤名的别名（短形式 → 标准名）
_STEP_ALIASES = {
    "prospectus": "prospectus_analyst",
    "debate": "debate_manager",
    "factcheck": "fact_check",
    "all": None,  # 特殊值
}


def normalize_step_names(steps: str | list[str]) -> list[str]:
    """把用户传的 steps 字符串解析成标准 agent name 列表，按执行顺序排好。"""
    if isinstance(steps, str):
        items = [s.strip() for s in steps.split(",") if s.strip()]
    else:
        items = list(steps)
    if not items or "all" in items:
        return sorted(_AGENT_REGISTRY.keys(), key=lambda n: _AGENT_REGISTRY[n][0])
    out: list[str] = []
    for s in items:
        canonical = _STEP_ALIASES.get(s, s)
        if canonical and canonical in _AGENT_REGISTRY and canonical not in out:
            out.append(canonical)
        elif canonical not in _AGENT_REGISTRY:
            raise ValueError(
                f"未知 step '{s}'。可选: {list(_AGENT_REGISTRY) + list(_STEP_ALIASES)}"
            )
    out.sort(key=lambda n: _AGENT_REGISTRY[n][0])
    return out


def rerun_steps(
    project_id: str,
    steps: str | list[str],
    *,
    llm: LLMClient | None = None,
    ticker: str | None = None,
    company_name: str | None = None,
    industry: str | None = None,
    peers: list[str] | None = None,
    recent_ipos: list[str] | None = None,
    ifind_target: str | None = None,
    prospectus_pdf: str | Path | None = None,
    debate_max_rounds: int | None = None,
) -> AgentContext:
    """重跑指定项目的某些 Agent 步骤，复用其他步骤已有的 brief。

    元数据来源优先级（缺失字段才看下一级）：
      函数显式参数 > reports_dir/_run_metadata.json > FeedbackStore predictions 表

    需要 RAG 的步骤（prospectus_analyst/industry/comparable/tech_trend）：
      复用 reports/<project_id> 对应的 ChromaDB collection。
      若未持久（CaseRAG 重命名 / 误删），从 prospectus_pdf 重建。

    需要 SDK 数据的步骤（industry/macro/comparable/sentiment）：
      用 peers/recent_ipos/ifind_target 重新调 _prefetch_ifind_sdk。
    """
    s = get_settings()
    reports_dir = s.reports_dir / project_id
    if not reports_dir.exists():
        raise FileNotFoundError(f"reports/{project_id}/ 不存在")

    # 1. 元数据合并
    metadata = load_run_metadata(reports_dir)
    pred_meta: dict[str, Any] = {}
    try:
        from src.feedback import FeedbackStore
        store = FeedbackStore()
        try:
            pred = store.get_prediction_by_project(project_id)
            if pred:
                pred_meta = {
                    "ticker": pred.ticker,
                    "company_name": pred.company_name,
                    "industry": pred.industry,
                }
        finally:
            store.close()
    except Exception as e:
        logger.debug(f"FeedbackStore 元数据读取跳过: {e}")

    def _pick(key: str, *fallbacks: Any) -> Any:
        for v in fallbacks:
            if v not in (None, "", []):
                return v
        return None

    ticker = _pick("ticker", ticker, metadata.get("ticker"), pred_meta.get("ticker"))
    company_name = _pick("company_name", company_name, metadata.get("company_name"),
                         pred_meta.get("company_name"))
    industry = _pick("industry", industry, metadata.get("industry"),
                     pred_meta.get("industry"))
    peers = _pick("peers", peers, metadata.get("peers"))
    recent_ipos = _pick("recent_ipos", recent_ipos, metadata.get("recent_ipos"))
    ifind_target = _pick("ifind_target", ifind_target, metadata.get("ifind_target"))
    prospectus_pdf = _pick("prospectus_pdf", prospectus_pdf, metadata.get("prospectus_pdf"))

    if not all([ticker, company_name, industry]):
        raise ValueError(
            "缺少 ticker / company_name / industry —— "
            "_run_metadata.json 不存在且 DB 无该 project，需要显式传参"
        )

    # 2. 解析 steps
    step_list = normalize_step_names(steps)
    if not step_list:
        logger.warning("rerun: steps 为空，无事可做")
        return AgentContext(  # type: ignore[call-arg]
            project_id=project_id, ticker=ticker, company_name=company_name,
            industry=industry, reports_dir=reports_dir,
        )
    needs_rag = any(_AGENT_REGISTRY[s][1] for s in step_list)
    needs_sdk = any(_AGENT_REGISTRY[s][2] for s in step_list)

    llm = llm or LLMClient()

    # 3. 构 ctx：RAG 复用（按需），extras 重建（按需 SDK 拉）
    rag: ProspectusRAG | None = None
    cached_blocks: list[str] = []
    if needs_rag:
        rag = ProspectusRAG(project_id=project_id)
        if not rag.is_indexed():
            if not prospectus_pdf or not Path(prospectus_pdf).exists():
                raise FileNotFoundError(
                    f"步骤 {step_list} 需要 RAG，但 ChromaDB 未索引且未提供 --pdf 重建。"
                )
            logger.info(f"重建 RAG: {prospectus_pdf}")
            chunks = ProspectusLoader(prospectus_pdf).load_chunks()
            rag.index(chunks)
            cached_blocks = ["\n\n---\n\n".join(select_cached_blocks(chunks))]
        else:
            logger.info(f"复用已有 RAG (chunks={rag._collection.count()})")  # type: ignore[union-attr]

    wf_extras = WorkflowExtras()
    if needs_sdk:
        # 复用 _prefetch_ifind_sdk 拉 peers / recent_ipos / target 数据
        if peers or recent_ipos:
            try:
                sdk_data = _prefetch_ifind_sdk(
                    list(peers or []), list(recent_ipos) if recent_ipos else None,
                    target_ticker=ifind_target or ticker,
                    expected_target_name=company_name,
                )
                wf_extras.peers = sdk_data["peers"]
                wf_extras.peer_recent_quotes = sdk_data["peer_recent_quotes"]
                wf_extras.recent_hk_ipos = sdk_data["recent_hk_ipos"]
                wf_extras.target_valuation = sdk_data.get("target_valuation")
                wf_extras.peer_announcements = sdk_data.get("peer_announcements", {})
                wf_extras.market_indices = sdk_data.get("market_indices", [])
                tv = sdk_data.get("target_valuation") or {}
                if tv.get("revenue"):
                    wf_extras.target_revenue = float(tv["revenue"])
                if tv.get("net_profit"):
                    wf_extras.target_net_profit = float(tv["net_profit"])
            except Exception as e:
                logger.warning(f"SDK prefetch 失败（继续，部分字段缺失）: {e}")
        # 宏观也通过 _prefetch_ths 拉 EDB
        try:
            ths_data = CornerstoneWorkflow._prefetch_ths(ticker, industry)
            for k, v in ths_data.items():
                wf_extras.set(k, v)
        except Exception as e:
            logger.warning(f"_prefetch_ths 失败（继续）: {e}")

    ctx = AgentContext(
        project_id=project_id,
        ticker=ticker,
        company_name=company_name,
        industry=industry,
        reports_dir=reports_dir,
        rag=rag,
        cached_blocks=cached_blocks,
        extras=wf_extras,
    )

    # 4. 加载已有 brief（不在重跑列表里的）
    import re as _re
    brief_pat = _re.compile(r"^(\d{2})_(.+)\.brief\.md$")
    for f in sorted(reports_dir.iterdir()):
        m = brief_pat.match(f.name)
        if not m:
            continue
        agent_name = m.group(2)
        if agent_name in step_list:
            continue
        ctx.briefs[agent_name] = f.read_text(encoding="utf-8")
    logger.info(
        f"rerun: 加载 {len(ctx.briefs)} 个已有 brief, "
        f"将重跑 {step_list}"
    )

    # 5. 实例化要跑的 agents (按步序号排)
    summarizer = Summarizer(llm)
    rounds_for_debate = (
        debate_max_rounds if debate_max_rounds is not None else get_settings().debate_max_rounds
    )

    def _instantiate(name: str) -> BaseAgent:
        from src.agents.bear import BearResearcher  # noqa: F401
        from src.agents.bull import BullResearcher  # noqa: F401
        from src.agents.fact_check import FactCheckerAgent
        klass_map = {
            "prospectus_analyst": ProspectusAnalystAgent,
            "industry": IndustryAgent,
            "macro": MacroAgent,
            "comparable": ComparableAgent,
            "tech_trend": TechTrendAgent,
            "sentiment": SentimentAgent,
            "fact_check": FactCheckerAgent,
            "risk": RiskAgent,
            "decision": DecisionAgent,
        }
        if name == "debate_manager":
            return DebateOrchestrator(llm, max_rounds=rounds_for_debate)
        klass = klass_map[name]
        if name in ("decision", "fact_check"):
            return klass(llm)  # FactCheckerAgent / DecisionAgent 不接 summarizer
        return klass(llm, summarizer)

    # 6. 跑各步骤
    for step_name in step_list:
        step_no = _AGENT_REGISTRY[step_name][0]
        agent = _instantiate(step_name)
        logger.info(f"--- rerun 步骤 {step_no}: {step_name} ---")
        report = agent.run(ctx)
        agent._save_full_report(ctx, step_no, report.full_report)
        (reports_dir / f"{step_no:02d}_{agent.name}.brief.md").write_text(
            report.brief, encoding="utf-8"
        )

    # 7. token 账本（独立文件，不覆盖原版）
    rerun_tag = "_".join(step_list)[:60]
    from src.llm.pricing import estimate_total_cost_cny
    from src.llm.router import ModelTier as _MT, resolve_model
    tier_to_model = {t.value: resolve_model(t) for t in _MT}
    cost = estimate_total_cost_cny(llm.ledger.by_tier, tier_to_model)
    (reports_dir / f"_token_usage_rerun_{rerun_tag}.md").write_text(
        f"# Token 账本（rerun: {step_list}）\n\n"
        + llm.ledger.summary()
        + f"\n\n**预估成本: ¥{cost}**\n",
        encoding="utf-8",
    )

    # 8. 决议被 rerun 时，更新 prediction（upsert）+ 重写 FINAL_MEMO
    if "decision" in step_list:
        try:
            persist_prediction(ctx, llm)
        except Exception as e:
            logger.warning(f"prediction 落库失败: {e}")
        try:
            from src.reports.writer import write_final_summary
            write_final_summary(ctx)
        except Exception as e:
            logger.warning(f"FINAL_MEMO 重写失败: {e}")

    return ctx


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    """安全读 obj.name, obj 为 None 或没有该属性时返默认值。"""
    if obj is None:
        return default
    return getattr(obj, name, default)


def persist_prediction(ctx: AgentContext, llm: LLMClient) -> int | None:
    """把 ctx 里的决议结果落到 feedback DB，返回 prediction id。

    决议缺失时返回 None。可被 Workflow 主流程或一次性脚本（rerun_decision）共用。
    """
    from datetime import datetime as _dt

    from src.feedback import FeedbackStore
    from src.feedback.models import Prediction
    from src.llm.pricing import estimate_total_cost_cny
    from src.llm.router import ModelTier, resolve_model

    decision = ctx.extras.decision_json
    if not decision:
        logger.info("决议 JSON 缺失，跳过 prediction 落库")
        return None

    amount = decision.get("suggested_amount_usd_million") or [0, 0]
    valuation = decision.get("valuation_range_hkd_billion") or {}

    agg = llm.ledger.by_tier
    total_in = sum(s.get("input", 0) for s in agg.values())
    total_out = sum(s.get("output", 0) for s in agg.values())
    total_cache = sum(s.get("cache_read", 0) for s in agg.values())
    tier_to_model = {t.value: resolve_model(t) for t in ModelTier}

    s = get_settings()

    score_cards = ctx.extras.misc.get("score_cards", {})
    features_used = ["scoring_card"]
    cog_features = ctx.extras.misc.get("cogalpha_features")
    if cog_features:
        features_used.extend(sorted(cog_features))

    p = Prediction(
        project_id=ctx.project_id,
        ticker=ctx.ticker,
        company_name=ctx.company_name,
        industry=ctx.industry,
        decision_date=_dt.now(),
        recommendation=str(decision.get("recommendation", "")),
        confidence=str(decision.get("confidence", "")),
        valuation_low=valuation.get("low"),
        valuation_mid=float(valuation.get("mid", 0) or 0),
        valuation_high=valuation.get("high"),
        anchor_method=str(valuation.get("anchor_method", "")),
        anchor_logic=str(valuation.get("anchor_logic", "")),
        ipo_pricing_view=str(decision.get("ipo_pricing_view", "")),
        suggested_amount_low_usd_m=float(amount[0]) if len(amount) > 0 else 0,
        suggested_amount_high_usd_m=float(amount[1]) if len(amount) > 1 else 0,
        key_supports=list(decision.get("key_supports", [])),
        key_risks=list(decision.get("key_risks", [])),
        deal_conditions=list(decision.get("deal_conditions", [])),
        monitoring_kpis=list(decision.get("monitoring_kpis", [])),
        # v4 决策因子加权打分卡, 持久化用于 PostmortemAgent 事后校准
        decision_weights=list(decision.get("decision_weights", []) or []),
        weighted_total_score=decision.get("weighted_total_score"),
        weighted_to_recommendation_mapping=str(
            decision.get("weighted_to_recommendation_mapping", "") or ""
        ),
        # v5 ListingProfile 核心字段, 用于 calibration priors 三维查询
        listing_chapter=_safe_attr(ctx.extras.listing_profile, "listing_chapter", "Unknown"),
        size_tier=_safe_attr(ctx.extras.listing_profile, "size_tier", "Unknown"),
        industry_theme=_safe_attr(ctx.extras.listing_profile, "industry_theme", "Other"),
        has_wvr=bool(_safe_attr(ctx.extras.listing_profile, "has_wvr", False)),
        has_a_share_listed=bool(_safe_attr(ctx.extras.listing_profile, "has_a_share_listed", False)),
        agent_score_cards=score_cards,
        model_provider=s.llm_provider,
        model_tier_models=tier_to_model,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cache_read_tokens=total_cache,
        estimated_cost_cny=estimate_total_cost_cny(agg, tier_to_model),
        cogalpha_features_used=features_used,
        reports_dir_path=str(ctx.reports_dir),
        status="open",
    )
    store = FeedbackStore()
    try:
        pid = store.save_prediction(p)
        logger.info(f"prediction 已落库 id={pid}")
        return pid
    finally:
        store.close()


class CornerstoneWorkflow:
    def __init__(self, llm: LLMClient | None = None, debate_max_rounds: int | None = None):
        self.llm = llm or LLMClient()
        self.summarizer = Summarizer(self.llm)
        s = get_settings()
        max_rounds = debate_max_rounds if debate_max_rounds is not None else s.debate_max_rounds

        from src.agents.fact_check import FactCheckerAgent
        self.steps: list[BaseAgent] = [
            ProspectusAnalystAgent(self.llm, self.summarizer),
            IndustryAgent(self.llm, self.summarizer),
            MacroAgent(self.llm, self.summarizer),
            ComparableAgent(self.llm, self.summarizer),
            TechTrendAgent(self.llm, self.summarizer),
            SentimentAgent(self.llm, self.summarizer),
            FactCheckerAgent(self.llm),  # ⭐ 新增: 跨 Agent 数字交叉核对
            DebateOrchestrator(self.llm, max_rounds=max_rounds),
            RiskAgent(self.llm, self.summarizer),
            DecisionAgent(self.llm),
        ]

    @staticmethod
    def _prefetch_ths(ticker: str, industry: str) -> dict:
        """跑 Agent 前先把 THS_BD/THS_EDB/THS_DR 数据预取一遍，丢到 ctx.extras。

        失败时返回空字典对应键，下游 Agent 自行处理空值。
        """
        ths = THSClient()
        if not ths.configured:
            logger.info("THS 未配置，跳过基础数据/宏观/研报预取")
            return {}

        code = ticker if "." in ticker else f"{ticker.zfill(5)}.HK"
        out: dict[str, Any] = {}

        try:
            bd = ths.basic_data(codes=code)
            out["company_basic"] = bd.get(code, {}) if isinstance(bd, dict) else {}
            logger.info(f"[Prefetch] basic_data 字段数: {len(out['company_basic'])}")
        except Exception as e:
            logger.warning(f"[Prefetch] basic_data 失败: {e}")
            out["company_basic"] = {}

        try:
            macro = ths.edb()
            out["macro_indicators"] = macro
            logger.info(f"[Prefetch] EDB 指标数: {len(macro)}")
        except Exception as e:
            logger.warning(f"[Prefetch] edb 失败: {e}")
            out["macro_indicators"] = {}

        try:
            reports = ths.research_reports(codes=code, industry=industry)
            out["industry_research"] = reports[:20]  # 限制条数控制 token
            logger.info(f"[Prefetch] 研报条数: {len(out['industry_research'])}")
        except Exception as e:
            logger.warning(f"[Prefetch] research_reports 失败: {e}")
            out["industry_research"] = []

        return out

    @staticmethod
    def _build_ctx(
        ticker: str,
        company_name: str,
        industry: str,
        prospectus_pdf: str | Path | None,
        extras: WorkflowExtras | None = None,
    ) -> AgentContext:
        s = get_settings()
        project_id = f"{ticker}_{datetime.now():%Y%m%d_%H%M%S}"
        reports_dir = s.reports_dir / project_id
        reports_dir.mkdir(parents=True, exist_ok=True)

        rag: ProspectusRAG | None = None
        cached_blocks: list[str] = []
        if prospectus_pdf:
            pdf_path = Path(prospectus_pdf)
            if pdf_path.exists():
                logger.info(f"加载招股书 {pdf_path}")
                chunks = ProspectusLoader(pdf_path).load_chunks()
                rag = ProspectusRAG(project_id=project_id)
                rag.index(chunks)
                # 按章节关键词挑选高信息密度章节作为 prompt cache 内容
                selected = select_cached_blocks(chunks)
                cached_blocks = ["\n\n---\n\n".join(selected)] if selected else []
                logger.info(
                    f"cached_blocks: {len(selected)} 个关键章节 / "
                    f"{sum(len(s) for s in selected)} 字符"
                )
            else:
                logger.warning(f"招股书 PDF 不存在: {pdf_path}")

        return AgentContext(
            project_id=project_id,
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            reports_dir=reports_dir,
            rag=rag,
            cached_blocks=cached_blocks,
            extras=extras or WorkflowExtras(),
        )

    def run(
        self,
        *,
        ticker: str,
        company_name: str,
        industry: str,
        prospectus_pdf: str | Path | None = None,
        extras: dict | None = None,
        use_case_rag: bool = True,
        peers: list[str] | None = None,
        peer_confirm_callback: Any = None,
        recent_ipos: list[str] | None = None,
        ifind_target: str | None = None,
        roadshow_signals: dict | None = None,
        competing_ipo_tickers: list[str] | None = None,
        peer_pool_keywords: list[str] | None = None,
        target_market_cap_hkd_b: float | None = None,
        peer_pool_cap_range: tuple[float, float] | None = None,
        use_peer_pool: bool = True,
        explicit_listing_profile: dict | None = None,
        use_listing_profile_detector: bool = True,
    ) -> AgentContext:
        # 1. 基础 prefetch (不依赖 peers)
        prefetched = self._prefetch_ths(ticker, industry)
        wf_extras = WorkflowExtras.from_dict({**(extras or {}), **prefetched})

        # 2. 建 ctx (含 RAG); peers 数据先空, 之后注入
        ctx = self._build_ctx(ticker, company_name, industry, prospectus_pdf, wf_extras)

        # 3. 交互式 peer 确认 (仅当未给 peers 且 callback 存在 且 RAG 可用)
        if not peers and peer_confirm_callback is not None and ctx.rag is not None:
            try:
                # v2: 先从 akshare 拉行业港股池, 让 LLM 从池子里选
                pool: list[dict] = []
                if use_peer_pool:
                    from src.data.peer_pool import build_peer_pool, derive_keywords_from_industry
                    kws = peer_pool_keywords or derive_keywords_from_industry(ctx.industry)
                    if kws:
                        try:
                            pool = build_peer_pool(
                                kws,
                                target_market_cap_hkd_b=target_market_cap_hkd_b,
                                explicit_cap_range=peer_pool_cap_range,
                            )
                            logger.info(
                                f"[PeerSuggester] 行业池构建完成: {len(pool)} 家 "
                                f"(关键词={kws})"
                            )
                        except Exception as e:
                            logger.warning(f"[PeerSuggester] 行业池构建失败: {e}")

                from src.agents.peer_suggester import suggest_peers
                candidates = suggest_peers(
                    ctx.rag, ctx.company_name, ctx.industry, self.llm,
                    pool=pool or None,
                    target_market_cap_hkd_b=target_market_cap_hkd_b,
                )
                logger.info(f"[PeerSuggester] LLM 候选 {len(candidates)} 个")
                peers = peer_confirm_callback(candidates) or None
            except Exception as e:
                logger.warning(f"[PeerSuggester] 失败, 跳过交互确认: {e}")

        # 把项目元数据落盘，供 rerun 命令复用（避免再让用户输 ticker/name/peers）
        save_run_metadata(
            ctx.reports_dir,
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            prospectus_pdf=str(prospectus_pdf) if prospectus_pdf else None,
            peers=list(peers) if peers else None,
            recent_ipos=list(recent_ipos) if recent_ipos else None,
            ifind_target=ifind_target,
            roadshow_signals=dict(roadshow_signals) if roadshow_signals else None,
            competing_ipo_tickers=list(competing_ipo_tickers) if competing_ipo_tickers else None,
            peer_pool_keywords=list(peer_pool_keywords) if peer_pool_keywords else None,
            target_market_cap_hkd_b=target_market_cap_hkd_b,
            explicit_listing_profile=dict(explicit_listing_profile) if explicit_listing_profile else None,
        )

        # 4. 用确认后的 peers + recent_ipos 拉 SDK 数据, setattr 到 ctx.extras
        # ifind_target 优先级: 显式传入 > project ticker(2670). 校验公司名防止代码污染。
        target_for_ifind = ifind_target or ticker
        if peers or recent_ipos:
            sdk_data = _prefetch_ifind_sdk(
                peers or [], recent_ipos,
                target_ticker=target_for_ifind,
                expected_target_name=company_name,
            )
            ctx.extras.peers = sdk_data["peers"]
            ctx.extras.peer_recent_quotes = sdk_data["peer_recent_quotes"]
            ctx.extras.recent_hk_ipos = sdk_data["recent_hk_ipos"]
            ctx.extras.target_valuation = sdk_data.get("target_valuation")
            ctx.extras.peer_announcements = sdk_data.get("peer_announcements", {})
            ctx.extras.market_indices = sdk_data.get("market_indices", [])
            # 同步 target 营收/净利到既有字段, 让 comparable_valuation 工具能用
            tv = sdk_data.get("target_valuation") or {}
            if tv.get("revenue"):
                ctx.extras.target_revenue = float(tv["revenue"])
            if tv.get("net_profit"):
                ctx.extras.target_net_profit = float(tv["net_profit"])

        # 路演手输信号 (#11) → ctx.extras.roadshow_signals
        if roadshow_signals:
            ctx.extras.roadshow_signals = dict(roadshow_signals)

        # 同期竞品 IPO (#14) → 拉每家的 IPO 信息存到 competing_ipos
        if competing_ipo_tickers:
            try:
                from src.data.ifind_sdk import get_peer_ipo_summary
                competing_list = []
                for t in competing_ipo_tickers:
                    info = get_peer_ipo_summary(t) or {}
                    if info:
                        competing_list.append({
                            "ticker": t,
                            "thscode": info.get("thscode"),
                            "name": info.get("name"),
                            "expected_listing_date": info.get("ipo_date") or "未确定",
                            "expected_marketcap_hkd_b": None,  # 招股阶段未确定
                            "ipo_price": info.get("ipo_price"),
                            "overlap_note": "同期同行业, 关注资金分流",
                        })
                ctx.extras.competing_ipos = competing_list
                logger.info(f"[Prefetch] competing_ipos: {len(competing_list)} 家")
            except Exception as e:
                logger.warning(f"[Prefetch] competing_ipos 失败: {e}")

        # 闭环关键 (Phase C): 检索历史相似案例，注入到 cached_blocks
        if use_case_rag:
            try:
                self._inject_similar_cases(ctx)
            except Exception as e:
                logger.warning(f"CaseRAG 检索失败（不影响主流程）: {e}")

        # 上市档案 ListingProfile: 显式 CLI > detector 推断 > Unknown 兜底
        try:
            self._build_listing_profile(
                ctx,
                explicit=explicit_listing_profile,
                use_detector=use_listing_profile_detector,
            )
        except Exception as e:
            logger.warning(f"ListingProfile 构建失败（不影响主流程）: {e}")

        # 决策因子权重校准 priors (Phase B): 同行业历史复盘 → 平均偏差 → 注入
        # 样本数 < 5 时优雅降级 (返回空, Decision prompt 不渲染)
        try:
            self._inject_weight_priors(ctx)
        except Exception as e:
            logger.warning(f"权重校准 priors 注入失败（不影响主流程）: {e}")

        logger.info(f"=== 工作流启动: {ctx.project_id} ===")

        for i, agent in enumerate(self.steps, start=1):
            fatal_tag = " (fatal)" if agent.fatal else ""
            logger.info(f"--- 步骤 {i}/{len(self.steps)}: {agent.name}{fatal_tag} ---")
            try:
                report = agent.run(ctx)
                agent._save_full_report(ctx, i, report.full_report)
                brief_path = ctx.reports_dir / f"{i:02d}_{agent.name}.brief.md"
                brief_path.write_text(report.brief, encoding="utf-8")
            except Exception as e:
                logger.exception(f"Agent {agent.name} 失败: {e}")
                err_path = ctx.reports_dir / f"{i:02d}_{agent.name}.ERROR.md"
                err_path.write_text(
                    f"# {agent.name} 执行失败\n\n```\n{type(e).__name__}: {e}\n```\n",
                    encoding="utf-8",
                )
                if agent.fatal:
                    # 写出 token 账本后再抛，至少留下成本痕迹
                    self._write_ledger(ctx)
                    logger.error(f"致命 Agent [{agent.name}] 失败，工作流中止")
                    raise
                # 非致命：注入失败标记到 brief，下游 LLM 能明确感知信息缺失
                ctx.briefs[agent.name] = (
                    f"**[执行失败]** 本环节因 `{type(e).__name__}` 异常未产出有效内容。\n"
                    f"信息缺失影响范围：{agent.description or agent.name}。\n"
                    f"请在你的分析与决策中**显式承认这部分信息空白**，"
                    f"不要凭空推断或假装拥有这部分数据。"
                )

        self._write_ledger(ctx)
        # 自动落库 Prediction（Phase A）
        try:
            self._persist_prediction(ctx)
        except Exception as e:
            logger.warning(f"prediction 落库失败（不影响报告产出）: {e}")
        logger.info(f"=== 工作流完成，报告目录: {ctx.reports_dir} ===")
        return ctx

    def _build_listing_profile(
        self,
        ctx: AgentContext,
        *,
        explicit: dict | None = None,
        use_detector: bool = True,
    ) -> None:
        """构造 ListingProfile, 注入 ctx.extras.listing_profile。

        优先级: 显式 CLI 参数 > detector 推断 > Unknown 兜底。
        显式给的字段不会被 detector 覆盖（用户输入最权威）。
        """
        from src.agents.listing_profile import ListingProfile

        explicit = explicit or {}
        # 行业主题缺失时尝试从 industry 字段推断
        if "industry_theme" not in explicit:
            explicit["industry_theme"] = self._guess_industry_theme(ctx.industry)

        # detector 仅当 listing_chapter / profitability_stage 等关键字段缺失时跑
        needs_detection = (
            "listing_chapter" not in explicit
            or "profitability_stage" not in explicit
        )

        detected: dict = {}
        if use_detector and needs_detection and ctx.rag is not None and ctx.rag.is_indexed():
            try:
                from src.agents.listing_profile_detector import detect_listing_profile
                detected = detect_listing_profile(ctx.rag, ctx.company_name, self.llm) or {}
            except ImportError:
                # detector 还没实现 (C2.5 之前), 静默跳过
                pass
            except Exception as e:
                logger.warning(f"ListingProfileDetector 失败: {e}")

        # 合并: 显式 > 推断 > 默认
        merged = {**detected, **explicit}
        if explicit and not detected:
            merged["detection_confidence"] = "显式确认"
        elif detected and not explicit:
            merged["detection_confidence"] = "招股书推断"
        elif merged.get("listing_chapter", "Unknown") != "Unknown":
            merged["detection_confidence"] = "显式确认"

        try:
            profile = ListingProfile.model_validate(merged)
        except Exception as e:
            logger.warning(f"ListingProfile 校验失败: {e}, 用 Unknown 兜底")
            profile = ListingProfile()
        ctx.extras.listing_profile = profile
        logger.info(
            f"[ListingProfile] chapter={profile.listing_chapter} / "
            f"stage={profile.profitability_stage} / size={profile.size_tier} / "
            f"theme={profile.industry_theme} / wvr={profile.has_wvr} / "
            f"a-share={profile.has_a_share_listed} / 置信={profile.detection_confidence}"
        )

    @staticmethod
    def _guess_industry_theme(industry: str) -> str:
        """从 industry 关键字粗略归类到 ListingProfile.industry_theme。"""
        if not industry:
            return "Other"
        low = industry.lower()
        if any(k in industry for k in ["生物", "医药", "制药", "Bio", "Pharma"]) or "pharm" in low:
            return "Bio_Pharma"
        if any(k in industry for k in ["医疗器械", "Med Device"]):
            return "Med_Device"
        if any(k in industry for k in ["机器人", "工业自动化", "协作", "Robotic"]):
            return "Robotics_Automation"
        if any(k in industry for k in ["AI", "人工智能", "半导体", "芯片", "Semi"]):
            return "Tech_AI_Semi"
        if any(k in industry for k in ["新能源", "光伏", "电池"]):
            return "New_Energy"
        if any(k in industry for k in ["材料"]):
            return "Advanced_Materials"
        if any(k in industry for k in ["消费", "零售", "餐饮", "服装"]):
            return "Consumer"
        if any(k in industry for k in ["金融", "银行", "保险", "证券"]):
            return "Financial"
        if any(k in industry for k in ["地产", "房地产", "物业"]):
            return "Real_Estate"
        if any(k in industry for k in ["医疗", "Health"]):
            return "Healthcare"
        if any(k in industry for k in ["工业", "制造", "Industrial"]):
            return "Industrial"
        return "Other"

    @staticmethod
    def _inject_weight_priors(ctx: AgentContext, min_samples: int = 5) -> None:
        """从历史 closed predictions 的 PostmortemAgent 输出聚合权重校准 priors,
        注入 ctx.extras.weight_priors. Decision Agent prompt 会按需渲染。

        v5: 三维度匹配 (industry + listing_chapter + size_tier), 自动 fallback。
        样本量 < min_samples 时不注入 (避免小样本误导)。
        """
        try:
            from src.feedback import FeedbackStore
        except ImportError:
            return
        # 从 ctx.extras.listing_profile 抽 chapter / size_tier
        profile = getattr(ctx.extras, "listing_profile", None)
        chapter = _safe_attr(profile, "listing_chapter", None)
        size_t = _safe_attr(profile, "size_tier", None)
        if chapter == "Unknown":
            chapter = None
        if size_t == "Unknown":
            size_t = None

        store = FeedbackStore()
        try:
            priors = store.get_weight_calibration_priors(
                industry=ctx.industry,
                recommendation=None,
                listing_chapter=chapter,
                size_tier=size_t,
                min_samples=min_samples,
            )
        finally:
            store.close()
        ctx.extras.weight_priors = priors
        n = priors.get("sample_size", 0)
        n_factors = len(priors.get("calibrations", []))
        if n_factors > 0:
            logger.info(
                f"[Weight Priors] 历史 {n} 个样本聚合 → {n_factors} 个因子的平均偏差, "
                f"将注入 Decision prompt"
            )
        else:
            logger.info(
                f"[Weight Priors] 历史 closed 样本 {n} (门槛 {min_samples}), 不注入校准"
            )

    @staticmethod
    def _inject_similar_cases(ctx: AgentContext) -> None:
        """从 CaseRAG 检索相似历史案例，把 prompt 文本拼到 ctx.cached_blocks 末尾。

        策略：找 top-3 相似案例（行业 + 估值规模 + 已有 outcome），
        渲染成 markdown 块，作为新的 cached_block 追加。
        所有 Agent 通过 cached_system_blocks 都能看到，但 Decision 受益最大。
        """
        from src.feedback import CaseRAG
        rag = CaseRAG()
        if rag.count() == 0:
            logger.info("CaseRAG 当前无历史案例，跳过注入")
            return
        # 估值规模未知（IPO 之前没决议），先按行业匹配
        valuation_hint = None
        if ctx.extras.peers:
            # 用同行平均规模作为粗略 hint
            mids = [
                p.get("market_cap_hkd_b")
                for p in ctx.extras.peers
                if isinstance(p, dict) and p.get("market_cap_hkd_b")
            ]
            if mids:
                valuation_hint = sum(mids) / len(mids)
        hits = rag.search(industry=ctx.industry, valuation_mid=valuation_hint, k=3)
        if not hits:
            return
        prompt_block = CaseRAG.format_for_prompt(hits)
        if prompt_block:
            ctx.cached_blocks.append(prompt_block)
            logger.info(f"注入 {len(hits)} 个历史相似案例到 cached_blocks")
            # 标记进 Prediction.cogalpha_features_used (后续 _persist_prediction 用)
            features = ctx.extras.misc.setdefault("cogalpha_features", set())
            if isinstance(features, set):
                features.add("case_rag")

    def _write_ledger(self, ctx: AgentContext) -> None:
        from src.llm.pricing import estimate_total_cost_cny
        from src.llm.router import ModelTier, resolve_model
        tier_to_model = {t.value: resolve_model(t) for t in ModelTier}
        cost = estimate_total_cost_cny(self.llm.ledger.by_tier, tier_to_model)
        (ctx.reports_dir / "_token_usage.md").write_text(
            "# Token 使用账本\n\n"
            + self.llm.ledger.summary()
            + f"\n\n**预估成本: ¥{cost}**\n\n"
            + "Tier → Model:\n"
            + "\n".join(f"- {t}: `{m}`" for t, m in tier_to_model.items()),
            encoding="utf-8",
        )

    def _persist_prediction(self, ctx: AgentContext) -> None:
        persist_prediction(ctx, self.llm)
