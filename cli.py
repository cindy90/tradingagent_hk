"""CLI 入口。

用法:
    python cli.py analyze --ticker 09999 --name "示例科技" --industry "AI/SaaS" --pdf data/prospectus/09999.pdf

可选参数:
    --debate-rounds N        覆盖默认辩论轮数
    --no-prospectus          跳过招股书加载（快速测试）
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from config import get_settings
from src.data.hkex_client import download_document
from src.data.ths_client import THSClient
from src.feedback import CaseRAG, FeedbackStore, Outcome, PostmortemAgent, reindex_all_cases
from src.graph import CornerstoneWorkflow
from src.graph.workflow import rerun_steps
from src.reports import write_final_summary

app = typer.Typer(add_completion=False, help="港股 IPO 基石投资分析工具")
console = Console()


def _setup_logging() -> None:
    s = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=s.log_level)


@app.command()
def analyze(
    ticker: str = typer.Option(..., "--ticker", "-t", help="港股代码，5 位数字，例如 09999"),
    name: str = typer.Option(..., "--name", "-n", help="公司中文名"),
    industry: str = typer.Option(..., "--industry", "-i", help="所属行业关键字"),
    pdf: Path | None = typer.Option(None, "--pdf", "-p", help="本地招股书 PDF 路径"),
    pdf_url: str | None = typer.Option(None, "--pdf-url", help="招股书 PDF 公开 URL（如 HKEX 链接），自动下载到本地"),
    debate_rounds: int | None = typer.Option(None, "--debate-rounds", help="辩论最大轮数"),
    no_prospectus: bool = typer.Option(False, "--no-prospectus", help="跳过招股书"),
    no_auto_fetch: bool = typer.Option(False, "--no-auto-fetch", help="禁用同花顺自动拉取招股书"),
    fetch_prefer: str = typer.Option("PHIP", "--fetch-prefer", help="自动拉取优先标题关键词: PHIP/Application/Prospectus"),
    peers: str | None = typer.Option(
        None,
        "--peers",
        help="可比公司港股代码列表，逗号分隔（4 或 5 位均可），如 '02432,09880,09660'。"
        "提供后会用 iFinD SDK 拉这些公司近 90 天 K 线 + IPO 信息，填到 sentiment / comparable agent 的输入。",
    ),
    no_confirm_peers: bool = typer.Option(
        False,
        "--no-confirm-peers",
        help="跳过交互式 peer 确认环节（默认会用 LLM 从招股书提取候选并让用户确认）。"
        "传 --peers 时本标志无效（已显式指定）。",
    ),
    recent_ipos: str | None = typer.Option(
        None,
        "--recent-ipos",
        help="近期同行业 IPO 港股代码列表，逗号分隔。用于 sentiment 第三节"
        "“近期同行业 IPO 暗盘/首日表现”。不传时默认 = peers（同一组）。",
    ),
    ifind_target: str | None = typer.Option(
        None,
        "--ifind-target",
        help="目标公司在 iFinD 的真实代码（招股阶段是副牌, 如 H2254 表示珞石）。"
        "不传时用 --ticker；但招股期间港股代码会被已上市公司复用（如 2670.HK 实际是云迹），"
        "代码内置公司名校验, 不匹配会自动跳过 target 数据。",
    ),
    # 路演信号（手输, iFinD 不一定能拿到）
    dark_pool_price: float | None = typer.Option(
        None, "--dark-pool-price",
        help="暗盘价（HKD），从富途/老虎手抄。sentiment 用",
    ),
    ipo_price_low: float | None = typer.Option(
        None, "--ipo-price-low",
        help="招股价区间下限（HKD），用于算暗盘溢价",
    ),
    oversubscribe_retail: float | None = typer.Option(
        None, "--oversubscribe-retail",
        help="散户超额认购倍数（如 80 表示 80 倍）",
    ),
    oversubscribe_intl: float | None = typer.Option(
        None, "--oversubscribe-intl",
        help="国际配售超额认购倍数",
    ),
    press_coverage: int | None = typer.Option(
        None, "--press-coverage", min=1, max=5,
        help="媒体覆盖热度 1-5 分（5=深度报道密集）",
    ),
    competing_ipos: str | None = typer.Option(
        None, "--competing-ipos",
        help="未来 60 天同行业 IPO 队列代码列表（逗号分隔），用于资金分流分析",
    ),
    # peer pool v2 控制
    peer_pool_keywords: str | None = typer.Option(
        None, "--peer-pool-keywords",
        help="行业港股池过滤关键词（逗号分隔, 任一命中公司名即留）。"
        "不传时从 --industry 自动派生（如 '工业机器人/协作' → ['机器人','工业','协作']）",
    ),
    target_market_cap: float | None = typer.Option(
        None, "--target-market-cap",
        help="目标公司估值（亿 HKD）, 用于行业池规模过滤（默认 0.2x-5x 区间）+ "
        "PeerSuggester 评估规模匹配度",
    ),
    peer_pool_cap_min: float | None = typer.Option(
        None, "--peer-pool-cap-min",
        help="行业池市值下限（亿 HKD）, 显式指定则覆盖 0.2x 自动区间",
    ),
    peer_pool_cap_max: float | None = typer.Option(
        None, "--peer-pool-cap-max",
        help="行业池市值上限（亿 HKD）, 显式指定则覆盖 5x 自动区间",
    ),
    no_peer_pool: bool = typer.Option(
        False, "--no-peer-pool",
        help="禁用行业池（退化到 v1: LLM 从招股书 RAG 自由提取, 不推荐）",
    ),
    # 上市档案 (ListingProfile) — 决定差异化权重 / 估值方法 / 风险维度
    listing_chapter: str | None = typer.Option(
        None, "--listing-chapter",
        help="HKEX 上市规则章节: Main_Board_Standard / Main_Board_18A (未盈利生物科技) / "
        "Main_Board_18C (特专科技) / Main_Board_19C (WVR) / "
        "Secondary_Listing (二次上市) / Dual_Primary_AH (AH 双重) / GEM. "
        "不传时由 detector 从招股书自动推断。",
    ),
    profitability_stage: str | None = typer.Option(
        None, "--profitability-stage",
        help="盈利状态: Profitable_Stable / Profitable_Growth / Pre_Profit_Late_Stage / "
        "Loss_Making_Growth / Pre_Commercial. 影响估值方法选择。",
    ),
    size_tier: str | None = typer.Option(
        None, "--size-tier",
        help="规模档 (亿 HKD): Small <30 / Mid 30-300 / Large 300-1000 / Mega >1000. "
        "不传时按 --target-market-cap 自动判断。",
    ),
    industry_theme: str | None = typer.Option(
        None, "--industry-theme",
        help="行业主题: Tech_AI_Semi / Bio_Pharma / Med_Device / Robotics_Automation / "
        "New_Energy / Advanced_Materials / Consumer / Financial / Real_Estate / "
        "Industrial / Healthcare / Other",
    ),
    has_wvr: bool = typer.Option(
        False, "--has-wvr",
        help="是否同股不同权架构 (影响治理风险加权)",
    ),
    a_share_ticker: str | None = typer.Option(
        None, "--a-share-ticker",
        help="A 股代码（如已 A 股上市，例: 688256）, 触发 A-H 折价估值锚定",
    ),
    main_listing_market: str | None = typer.Option(
        None, "--main-listing-market",
        help="二次上市的主上市地, 例: NASDAQ / NYSE",
    ),
    no_html: bool = typer.Option(
        False, "--no-html",
        help="禁用 FINAL_MEMO.html 渲染（默认同时输出 .md + .html）",
    ),
    open_html: bool = typer.Option(
        False, "--open-html",
        help="生成完成后自动用浏览器打开 FINAL_MEMO.html",
    ),
) -> None:
    """对单个 IPO 项目跑完整深度分析。"""
    _setup_logging()
    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower()
    key_map = {"anthropic": s.anthropic_api_key, "kimi": s.kimi_api_key, "deepseek": s.deepseek_api_key}
    if not key_map.get(provider):
        console.print(f"[red]LLM_PROVIDER={provider} 但对应 API Key 未配置[/red]")
        raise typer.Exit(code=1)

    pdf_arg: Path | None = None
    if not no_prospectus:
        # 优先级: --pdf > --pdf-url > 本地缓存 > 同花顺自动拉取
        pdf_arg = pdf or _try_default_pdf(ticker)
        if pdf_arg is None and pdf_url:
            local = s.prospectus_dir / f"{ticker}.pdf"
            console.print(f"[cyan]从 URL 下载招股书: {pdf_url}[/cyan]")
            if download_document(pdf_url, str(local)):
                pdf_arg = local
                console.print(f"[green]✓ 已下载: {pdf_arg}[/green]")
            else:
                console.print("[yellow]URL 下载失败，将尝试同花顺[/yellow]")
        if pdf_arg is None and not no_auto_fetch:
            ths = THSClient()
            if ths.configured:
                console.print(f"[cyan]本地未找到招股书，尝试通过同花顺拉取 {ticker} ...[/cyan]")
                pdf_arg = ths.auto_fetch_prospectus(ticker, prefer=fetch_prefer)
                if pdf_arg:
                    console.print(f"[green]✓ 招股书已下载: {pdf_arg}[/green]")
                else:
                    console.print("[yellow]同花顺未返回可用招股书，将在无 RAG 模式下运行[/yellow]")
            else:
                console.print("[yellow]THS_REFRESH_TOKEN 未配置，跳过自动获取招股书[/yellow]")

    if pdf_arg and not Path(pdf_arg).exists():
        console.print(f"[yellow]招股书 PDF 不存在: {pdf_arg}，将跳过 RAG[/yellow]")
        pdf_arg = None

    peer_list: list[str] | None = None
    if peers:
        peer_list = [p.strip() for p in peers.split(",") if p.strip()]
        console.print(f"[cyan]可比公司输入: {peer_list}[/cyan]")

    recent_ipos_list: list[str] | None = None
    if recent_ipos:
        recent_ipos_list = [p.strip() for p in recent_ipos.split(",") if p.strip()]
        console.print(f"[cyan]近期 IPO 队列: {recent_ipos_list}[/cyan]")
    elif peer_list:
        # peer 同时被 sentiment 当成"近期同行业 IPO"样本; 业务上这两组应不同。
        # peers = 估值最像的(可能 5 年前上市); recent_ipos = 近 6 月新上市的(反映打新情绪)
        console.print(
            "[yellow]⚠ 未传 --recent-ipos，sentiment 第三节将 fallback 到 --peers 同一组。"
            "建议另传一组近 6 月港股新股代码以反映真实打新情绪 (例: "
            "--recent-ipos 09080,06699,09660)。[/yellow]"
        )

    callback = None if (peer_list or no_confirm_peers) else _interactive_peer_confirm

    # 路演手输信号 → ctx.extras.roadshow_signals
    roadshow_signals: dict = {}
    if dark_pool_price is not None:
        roadshow_signals["dark_pool_price"] = dark_pool_price
    if ipo_price_low is not None:
        roadshow_signals["ipo_price_low"] = ipo_price_low
    if oversubscribe_retail is not None:
        roadshow_signals["oversubscribe_retail_x"] = oversubscribe_retail
    if oversubscribe_intl is not None:
        roadshow_signals["oversubscribe_intl_x"] = oversubscribe_intl
    if press_coverage is not None:
        roadshow_signals["press_coverage_score"] = press_coverage

    competing_ipos_list: list[str] | None = None
    if competing_ipos:
        competing_ipos_list = [p.strip() for p in competing_ipos.split(",") if p.strip()]

    peer_pool_kw_list: list[str] | None = None
    if peer_pool_keywords:
        peer_pool_kw_list = [k.strip() for k in peer_pool_keywords.split(",") if k.strip()]
    cap_range: tuple[float, float] | None = None
    if peer_pool_cap_min is not None or peer_pool_cap_max is not None:
        cap_range = (peer_pool_cap_min, peer_pool_cap_max)  # type: ignore[assignment]

    # ListingProfile 显式输入 (CLI 优先, detector 兜底)
    explicit_profile_kwargs: dict = {}
    if listing_chapter:
        explicit_profile_kwargs["listing_chapter"] = listing_chapter
    if profitability_stage:
        explicit_profile_kwargs["profitability_stage"] = profitability_stage
    if size_tier:
        explicit_profile_kwargs["size_tier"] = size_tier
    elif target_market_cap is not None:
        # 按 target_market_cap 自动归档
        if target_market_cap < 30:
            explicit_profile_kwargs["size_tier"] = "Small"
        elif target_market_cap < 300:
            explicit_profile_kwargs["size_tier"] = "Mid"
        elif target_market_cap < 1000:
            explicit_profile_kwargs["size_tier"] = "Large"
        else:
            explicit_profile_kwargs["size_tier"] = "Mega"
    if industry_theme:
        explicit_profile_kwargs["industry_theme"] = industry_theme
    if has_wvr:
        explicit_profile_kwargs["has_wvr"] = True
    if a_share_ticker:
        explicit_profile_kwargs["has_a_share_listed"] = True
        explicit_profile_kwargs["a_share_ticker"] = a_share_ticker
    if main_listing_market:
        explicit_profile_kwargs["main_listing_market"] = main_listing_market
        if not listing_chapter:
            explicit_profile_kwargs["listing_chapter"] = "Secondary_Listing"

    workflow = CornerstoneWorkflow(debate_max_rounds=debate_rounds)
    ctx = workflow.run(
        ticker=ticker,
        company_name=name,
        industry=industry,
        roadshow_signals=roadshow_signals if roadshow_signals else None,
        competing_ipo_tickers=competing_ipos_list,
        explicit_listing_profile=explicit_profile_kwargs or None,
        peer_pool_keywords=peer_pool_kw_list,
        target_market_cap_hkd_b=target_market_cap,
        peer_pool_cap_range=cap_range,
        use_peer_pool=not no_peer_pool,
        prospectus_pdf=pdf_arg,
        peers=peer_list,
        peer_confirm_callback=callback,
        recent_ipos=recent_ipos_list,
        ifind_target=ifind_target,
    )

    final_path = write_final_summary(ctx, also_html=not no_html)
    console.print(f"\n[green]✓ 完成。投决备忘录 (Markdown):[/green] {final_path}")
    if not no_html:
        html_path = ctx.reports_dir / "FINAL_MEMO.html"
        if html_path.exists():
            console.print(f"[green]  投决备忘录 (HTML):[/green] {html_path}")
            if open_html:
                import webbrowser
                webbrowser.open(html_path.as_uri())
                console.print(f"[cyan]  已尝试用浏览器打开[/cyan]")
    console.print(f"[green]  分项报告目录:[/green] {ctx.reports_dir}")
    decision = ctx.extras.decision_json
    if decision:
        console.print("\n[bold]核心决议:[/bold]")
        console.print(decision)


def _interactive_peer_confirm(candidates: list) -> list[str]:
    """命令行交互: 展示 LLM 候选 peers, 让用户确认/编辑。

    Args:
        candidates: list[PeerCandidate] (含 ticker/name/reason)

    Returns: 5 位 ticker 字符串列表; 空列表 = 不用 peer。
    """
    from rich.table import Table

    if not candidates:
        console.print("[yellow]LLM 未从招股书提取到可比 peers, 你可以手工输入或跳过。[/yellow]")
        ans = typer.prompt(
            "输入要用的港股代码 (逗号分隔, 如 '02432,01021'); 直接回车=不用 peer",
            default="",
            show_default=False,
        )
    else:
        table = Table(title="LLM 从行业港股池提取的可比公司候选 (按业务相似度降序)")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("代码", style="green")
        table.add_column("简称")
        table.add_column("相似度", style="yellow")
        table.add_column("市值(亿HKD)", style="dim")
        table.add_column("理由")
        for i, c in enumerate(candidates, 1):
            score_str = f"{c.similarity_score:.1f}"
            cap_str = (
                f"{c.market_cap_hkd_b:.1f}" if c.market_cap_hkd_b is not None else "—"
            )
            table.add_row(str(i), c.ticker, c.name, score_str, cap_str, c.reason)
        console.print(table)
        console.print(
            "[bold]操作:[/bold] 直接回车=采用全部 / 输入逗号分隔代码=替换 / "
            "输入 [italic]'+02432,01021'[/italic]=追加 / 输入 [italic]'skip'[/italic]=不用 peer"
        )
        default_csv = ",".join(c.ticker for c in candidates)
        ans = typer.prompt("你的选择", default=default_csv, show_default=False)
    ans = ans.strip()
    if ans.lower() == "skip" or ans == "":
        return []
    if ans.startswith("+"):
        # 追加
        existing = [c.ticker for c in candidates]
        adds = [s.strip() for s in ans[1:].split(",") if s.strip()]
        return existing + [a for a in adds if a not in existing]
    # 替换
    return [s.strip() for s in ans.split(",") if s.strip()]


def _try_default_pdf(ticker: str) -> Path | None:
    s = get_settings()
    for ext in (".pdf", ".PDF"):
        p = s.prospectus_dir / f"{ticker}{ext}"
        if p.exists():
            return p
    return None


@app.command("record-outcome")
def record_outcome_cmd(
    project_id: str = typer.Option(..., "--project-id", "-p"),
    ipo_price: float | None = typer.Option(None, "--ipo-price", help="招股价 HKD"),
    ipo_marketcap_b: float | None = typer.Option(None, "--ipo-marketcap-b", help="市值（亿港元）"),
    d1: float | None = typer.Option(None, "--d1", help="首日涨跌幅小数（0.15 = +15%）"),
    d30: float | None = typer.Option(None, "--d30"),
    d90: float | None = typer.Option(None, "--d90"),
    d180: float | None = typer.Option(None, "--d180", help="6 月禁售期满收益"),
    d365: float | None = typer.Option(None, "--d365"),
    broken_ipo_d1: bool | None = typer.Option(None, "--broken-ipo-d1/--no-broken-ipo-d1"),
    broken_ipo_d180: bool | None = typer.Option(None, "--broken-ipo-d180/--no-broken-ipo-d180"),
    max_dd_lockup: float | None = typer.Option(None, "--max-dd-lockup", help="锁定期最大回撤 %"),
    notes: str = typer.Option("", "--notes", "-n"),
    close: bool = typer.Option(False, "--close", help="同时把 prediction 状态置为 closed"),
    auto_fetch: bool = typer.Option(
        False,
        "--auto-fetch",
        help="给定 --ipo-price 后从 iFinD 自动拉历史 K 线计算 d1/d30/d90/d180/d365 等所有 returns。"
        " 显式传的 --d1/--d30/... 优先级高于自动值。",
    ),
    listing_date: str | None = typer.Option(
        None, "--listing-date", help="上市日期 YYYY-MM-DD（auto-fetch 时若 iFinD 拉不到可显式传）"
    ),
    ticker: str | None = typer.Option(
        None, "--ticker", help="auto-fetch 用，未传时从 prediction 读"
    ),
) -> None:
    """录入实际投后表现，关联到指定 project_id。

    --auto-fetch 模式（推荐）：
        给定 --ipo-price 后，从 iFinD 自动拉历史 K 线，算出 d1/d30/d90/d180/d365 收益、
        是否破发、锁定期最大回撤、180 日均成交。用户只需手工填 --notes / 重大事件。
    """
    _setup_logging()
    store = FeedbackStore()
    pred = store.get_prediction_by_project(project_id)
    if pred is None:
        console.print(f"[red]未找到 project_id={project_id} 的 prediction[/red]")
        raise typer.Exit(code=2)

    pred_row = store._conn.execute(
        "SELECT id FROM predictions WHERE project_id=?", (project_id,)
    ).fetchone()
    pred_id = pred_row["id"]

    # auto-fetch: 从 iFinD 自动算 returns
    auto: dict = {}
    if auto_fetch:
        if ipo_price is None:
            console.print("[red]--auto-fetch 需要至少 --ipo-price[/red]")
            raise typer.Exit(code=2)
        eff_ticker = ticker or pred.ticker
        try:
            from src.data.ifind_sdk import compute_post_ipo_returns
            console.print(
                f"[cyan]正在从 iFinD 拉 {eff_ticker} 上市后股价（招股价 {ipo_price} HKD）...[/cyan]"
            )
            auto = compute_post_ipo_returns(
                ticker=eff_ticker,
                ipo_price=ipo_price,
                listing_date=listing_date,
            )
            if auto.get("error"):
                console.print(f"[red]iFinD 拉取失败: {auto['error']}[/red]")
                raise typer.Exit(code=2)
            console.print(f"[green]✓ 自动 fetch 完成，覆盖未显式传的字段[/green]")
            console.print(
                f"  上市日: {auto.get('listing_date')}, "
                f"D1: {_pct(auto.get('d1_return'))}, "
                f"D180: {_pct(auto.get('d180_return'))}, "
                f"D365: {_pct(auto.get('d365_return'))}, "
                f"破发(D180): {auto.get('was_broken_d180')}, "
                f"最大回撤: {auto.get('max_drawdown_in_d180_pct')}%"
            )
        except Exception as e:
            console.print(f"[red]auto-fetch 异常: {e}[/red]")
            raise typer.Exit(code=2)

    # 用户显式传 > auto > None
    def _pick(user_v, auto_key):
        return user_v if user_v is not None else auto.get(auto_key)

    outcome = Outcome(
        prediction_id=pred_id,
        recorded_date=datetime.now(),
        ipo_actual_price_hkd=ipo_price,
        ipo_actual_marketcap_hkd_billion=ipo_marketcap_b,
        d1_return=_pick(d1, "d1_return"),
        d30_return=_pick(d30, "d30_return"),
        d90_return=_pick(d90, "d90_return"),
        d180_return=_pick(d180, "d180_return"),
        d365_return=_pick(d365, "d365_return"),
        was_broken_ipo_d1=_pick(broken_ipo_d1, "was_broken_d1"),
        was_broken_ipo_d180=_pick(broken_ipo_d180, "was_broken_d180"),
        max_drawdown_in_lockup_pct=_pick(max_dd_lockup, "max_drawdown_in_d180_pct"),
        avg_daily_turnover_hkd_m_d180=auto.get("avg_daily_turnover_hkd_m_d180"),
        final_listing_date=_parse_iso_date(auto.get("listing_date")),
        user_notes=notes,
    )
    oid = store.record_outcome(outcome)
    console.print(f"[green]✓ outcome id={oid} 已录入[/green]")
    if close:
        store.update_status(pred_id, "closed")
        console.print(f"[green]  prediction id={pred_id} 状态置为 closed[/green]")


def _pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _parse_iso_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        from datetime import date as _date
        return _date.fromisoformat(s[:10])
    except Exception:
        return None


@app.command("list-predictions")
def list_predictions_cmd(
    status: str = typer.Option("", "--status", help="open | closed | abandoned；空则全部"),
    ticker: str = typer.Option("", "--ticker", "-t"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """列出所有投决记录及关键字段。"""
    _setup_logging()
    store = FeedbackStore()
    preds = store.list_predictions(status=status or None, ticker=ticker or None, limit=limit)
    if not preds:
        console.print("[yellow](暂无记录)[/yellow]")
        return
    from rich.table import Table
    t = Table(show_lines=False, header_style="bold cyan")
    for col in ["项目ID", "代码", "公司", "建议", "置信", "估值中枢", "状态", "决策日"]:
        t.add_column(col)
    for p in preds:
        t.add_row(
            p.project_id, p.ticker, p.company_name[:20],
            p.recommendation or "—", p.confidence or "—",
            f"{p.valuation_mid:.1f}" if p.valuation_mid else "—",
            p.status, p.decision_date.strftime("%Y-%m-%d"),
        )
    console.print(t)


@app.command("export-predictions")
def export_predictions_cmd(
    out_path: Path = typer.Option(Path("./reports/predictions_export.csv"), "--out", "-o"),
) -> None:
    """导出 predictions × outcomes × scores 的 join 视图为 CSV。"""
    _setup_logging()
    import csv
    store = FeedbackStore()
    rows = store.export_predictions_with_outcomes()
    if not rows:
        console.print("[yellow](无数据可导出)[/yellow]")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]✓ 导出 {len(rows)} 行 → {out_path}[/green]")


@app.command()
def stats() -> None:
    """投决归档统计概览。"""
    _setup_logging()
    store = FeedbackStore()
    s = store.stats_summary()
    rag = CaseRAG()
    try:
        case_count = rag.count()
    except Exception:
        case_count = 0
    console.print(f"[bold cyan]投决归档统计[/bold cyan]")
    for k, v in s.items():
        console.print(f"  {k}: {v}")
    console.print(f"  case_rag_indexed: {case_count}")


@app.command()
def review(
    project_id: str = typer.Option(..., "--project-id", "-p"),
) -> None:
    """对指定 project 跑 PostmortemAgent，输出复盘 memo + 落库 Score + 索引到 CaseRAG。"""
    _setup_logging()
    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower()
    key_map = {"anthropic": s.anthropic_api_key, "kimi": s.kimi_api_key, "deepseek": s.deepseek_api_key}
    if not key_map.get(provider):
        console.print(f"[red]LLM_PROVIDER={provider} 但对应 API Key 未配置[/red]")
        raise typer.Exit(code=1)

    store = FeedbackStore()
    pred = store.get_prediction_by_project(project_id)
    if pred is None:
        console.print(f"[red]未找到 project_id={project_id}[/red]")
        raise typer.Exit(code=2)
    pred_row = store._conn.execute(
        "SELECT id FROM predictions WHERE project_id=?", (project_id,)
    ).fetchone()
    pred_id = pred_row["id"]

    if store.latest_outcome(pred_id) is None:
        console.print(f"[yellow]该 prediction 尚未录入 outcome，无法复盘[/yellow]")
        console.print(f"先运行: ta-hk record-outcome -p {project_id} ...")
        raise typer.Exit(code=3)

    console.print(f"[cyan]开始复盘 {project_id} ...[/cyan]")
    score = PostmortemAgent().run(pred_id, store=store)
    if score is None:
        console.print(f"[red]复盘失败[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ 复盘完成[/green]")
    console.print(f"  recommendation_score: {score.recommendation_score:.2f}")
    console.print(f"  valuation_within_range: {score.valuation_within_range}")
    console.print(f"  valuation_error_pct: {score.valuation_error_pct}")
    console.print(f"  风险命中: {score.risks_realized_count}/{score.risks_total_count}, "
                  f"未识别风险: {score.unforeseen_risks_count}")
    console.print(f"  根因: {score.error_root_causes}")
    if pred.reports_dir_path:
        console.print(f"\n  完整复盘 memo: {pred.reports_dir_path}/POSTMORTEM.md")


@app.command()
def similar(
    industry: str = typer.Option(..., "--industry", "-i"),
    valuation_mid: float | None = typer.Option(None, "--valuation-mid", help="目标公司估值中枢 亿港元"),
    k: int = typer.Option(3, "--k"),
) -> None:
    """检索 CaseRAG 里行业/规模相近的历史案例（用于排查 case 注入是否生效）。"""
    _setup_logging()
    rag = CaseRAG()
    if rag.count() == 0:
        console.print("[yellow]CaseRAG 当前无案例。先 record-outcome + review 几个项目让它积累[/yellow]")
        return
    hits = rag.search(industry=industry, valuation_mid=valuation_mid, k=k)
    if not hits:
        console.print("[yellow]未检索到匹配案例[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        md = h["metadata"]
        console.print(f"[bold cyan]案例 {i}: {md.get('ticker')} ({md.get('industry')})[/bold cyan]")
        console.print(f"  估值中枢 {md.get('valuation_mid')} 亿 / 决议 {md.get('recommendation')} "
                      f"/ D180 {md.get('d180_return', 0) * 100:.1f}%")
        console.print(f"  距离: {h.get('distance')}")
        console.print(f"  ─── 摘要 ───\n{h['text'][:600]}\n")


@app.command()
def rerun(
    project_id: str = typer.Option(..., "--project-id", "-p", help="reports/<project_id>/ 必须存在"),
    steps: str = typer.Option(
        "decision",
        "--steps",
        "-s",
        help="逗号分隔的 agent 名: prospectus / industry / macro / comparable / "
        "tech_trend / sentiment / debate / risk / decision；'all' 重跑全部。",
    ),
    ticker: str | None = typer.Option(None, "--ticker", "-t",
                                      help="未传时优先读 _run_metadata.json，再读 DB"),
    name: str | None = typer.Option(None, "--name", "-n"),
    industry: str | None = typer.Option(None, "--industry", "-i"),
    pdf: Path | None = typer.Option(None, "--pdf", help="RAG 重建用（chroma 已索引时可省）"),
    peers: str | None = typer.Option(None, "--peers"),
    recent_ipos: str | None = typer.Option(None, "--recent-ipos"),
    ifind_target: str | None = typer.Option(None, "--ifind-target"),
    debate_rounds: int | None = typer.Option(None, "--debate-rounds"),
) -> None:
    """重跑指定项目的某些 Agent 步骤，复用其他步骤已有的 brief。

    最常用：决议步骤超时或想换 prompt 重生成
        ta-hk rerun -p 02670_20260509_104530 -s decision

    重跑多步：
        ta-hk rerun -p ... -s comparable,decision

    全量重跑（少用，相当于 analyze）：
        ta-hk rerun -p ... -s all
    """
    _setup_logging()
    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower()
    key_map = {"anthropic": s.anthropic_api_key, "kimi": s.kimi_api_key, "deepseek": s.deepseek_api_key}
    if not key_map.get(provider):
        console.print(f"[red]LLM_PROVIDER={provider} 但对应 API Key 未配置[/red]")
        raise typer.Exit(code=1)

    peer_list = [p.strip() for p in peers.split(",") if p.strip()] if peers else None
    recent_ipos_list = [p.strip() for p in recent_ipos.split(",") if p.strip()] if recent_ipos else None

    try:
        ctx = rerun_steps(
            project_id=project_id,
            steps=steps,
            ticker=ticker,
            company_name=name,
            industry=industry,
            peers=peer_list,
            recent_ipos=recent_ipos_list,
            ifind_target=ifind_target,
            prospectus_pdf=pdf,
            debate_max_rounds=debate_rounds,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)

    console.print(f"\n[green]✓ rerun 完成[/green]")
    console.print(f"  报告目录: {ctx.reports_dir}")
    if ctx.extras.decision_json:
        console.print("\n[bold]核心决议:[/bold]")
        console.print(ctx.extras.decision_json)


@app.command("suggest-peers")
def suggest_peers_cmd(
    project_id: str = typer.Option(..., "--project-id", "-p"),
    name: str | None = typer.Option(None, "--name"),
    industry: str | None = typer.Option(None, "--industry"),
) -> None:
    """对已有 reports/<project_id>/ 项目跑 PeerSuggester（不跑 9 个 agent）。

    用于：调试 RAG 召回质量 / 验证 LLM 提取效果，不浪费整次分析的 token。
    """
    _setup_logging()
    from src.agents.peer_suggester import suggest_peers
    from src.data.rag import ProspectusRAG
    from src.graph.workflow import load_run_metadata
    from src.llm import LLMClient

    s = get_settings()
    reports_dir = s.reports_dir / project_id
    md = load_run_metadata(reports_dir)
    name = name or md.get("company_name")
    industry = industry or md.get("industry")
    if not name or not industry:
        console.print("[red]缺 --name / --industry（_run_metadata.json 也无）[/red]")
        raise typer.Exit(code=2)

    rag = ProspectusRAG(project_id=project_id)
    if not rag.is_indexed():
        console.print(f"[red]RAG project_id={project_id} 未索引[/red]")
        raise typer.Exit(code=2)

    candidates = suggest_peers(rag, name, industry, LLMClient())
    if not candidates:
        console.print("[yellow]LLM 未返回候选 peers[/yellow]")
        return
    from rich.table import Table
    t = Table(title=f"{name} 的 LLM 候选 peers")
    t.add_column("代码", style="green")
    t.add_column("简称")
    t.add_column("理由")
    for c in candidates:
        t.add_row(c.ticker, c.name, c.reason)
    console.print(t)


@app.command("reindex-cases")
def reindex_cases_cmd() -> None:
    """从 SQLite 全量重建 CaseRAG 索引（embedding 模型切换或库损坏后用）。"""
    _setup_logging()
    n = reindex_all_cases()
    console.print(f"[green]✓ 已重建 {n} 个案例[/green]")


@app.command()
def show_config() -> None:
    """打印当前配置。敏感字段自动掩码。"""
    s = get_settings()
    d = s.model_dump()
    for k in list(d.keys()):
        if any(t in k.lower() for t in ("api_key", "token", "password", "secret")):
            v = d[k]
            d[k] = (v[:6] + "***" + v[-4:]) if isinstance(v, str) and len(v) > 12 else "***"
    console.print(d)


@app.command()
def fetch_prospectus(
    ticker: str = typer.Option(..., "--ticker", "-t", help="港股代码"),
    prefer: str = typer.Option("PHIP", "--prefer"),
    out: Path | None = typer.Option(None, "--out", help="保存路径，不传则用默认目录"),
) -> None:
    """仅通过同花顺拉取招股书 PDF（独立命令，方便调试）。"""
    _setup_logging()
    ths = THSClient()
    if not ths.configured:
        console.print("[red]THS_REFRESH_TOKEN 未配置[/red]")
        raise typer.Exit(code=1)
    p = ths.auto_fetch_prospectus(ticker, dest_dir=out.parent if out else None, prefer=prefer)
    if p is None:
        console.print(f"[red]✗ 未能获取 {ticker} 的招股书[/red]")
        raise typer.Exit(code=2)
    console.print(f"[green]✓ {p}[/green]")


@app.command()
def list_announcements(
    ticker: str = typer.Option(..., "--ticker", "-t"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str | None = typer.Option(None, "--end"),
    keyword: str | None = typer.Option(None, "--keyword"),
) -> None:
    """列出某 ticker 的公告（用于排查招股书匹配问题）。"""
    _setup_logging()
    ths = THSClient()
    if not ths.configured:
        console.print("[red]THS_REFRESH_TOKEN 未配置[/red]")
        raise typer.Exit(code=1)
    code = ticker if "." in ticker else f"{ticker.zfill(5)}.HK"
    fp: dict = {"startDate": start}
    if end:
        fp["endDate"] = end
    if keyword:
        fp["keyword"] = keyword
    rows = ths.query_reports(codes=code, function_params=fp)
    if not rows:
        console.print("[yellow](无返回)[/yellow]")
        return
    for r in rows:
        date = r.get("declareDate") or r.get("DECLAREDATE") or r.get("publishTime") or ""
        title = r.get("title") or r.get("TITLE") or ""
        url = r.get("pdfURL") or r.get("PDFURL") or ""
        console.print(f"{date}  {title}\n  → {url}")


if __name__ == "__main__":
    app()
