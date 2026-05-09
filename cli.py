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
from src.feedback import FeedbackStore, Outcome
from src.graph import CornerstoneWorkflow
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

    workflow = CornerstoneWorkflow(debate_max_rounds=debate_rounds)
    ctx = workflow.run(
        ticker=ticker,
        company_name=name,
        industry=industry,
        prospectus_pdf=pdf_arg,
    )

    final_path = write_final_summary(ctx)
    console.print(f"\n[green]✓ 完成。投决备忘录:[/green] {final_path}")
    console.print(f"[green]  分项报告目录:[/green] {ctx.reports_dir}")
    decision = ctx.extras.decision_json
    if decision:
        console.print("\n[bold]核心决议:[/bold]")
        console.print(decision)


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
    d180: float | None = typer.Option(None, "--d180", help="6 月禁售期满收益"),
    d365: float | None = typer.Option(None, "--d365"),
    broken_ipo_d1: bool | None = typer.Option(None, "--broken-ipo-d1/--no-broken-ipo-d1"),
    broken_ipo_d180: bool | None = typer.Option(None, "--broken-ipo-d180/--no-broken-ipo-d180"),
    max_dd_lockup: float | None = typer.Option(None, "--max-dd-lockup", help="锁定期最大回撤 %"),
    notes: str = typer.Option("", "--notes", "-n"),
    close: bool = typer.Option(False, "--close", help="同时把 prediction 状态置为 closed"),
) -> None:
    """录入实际投后表现，关联到指定 project_id。"""
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

    outcome = Outcome(
        prediction_id=pred_id,
        recorded_date=datetime.now(),
        ipo_actual_price_hkd=ipo_price,
        ipo_actual_marketcap_hkd_billion=ipo_marketcap_b,
        d1_return=d1,
        d30_return=d30,
        d180_return=d180,
        d365_return=d365,
        was_broken_ipo_d1=broken_ipo_d1,
        was_broken_ipo_d180=broken_ipo_d180,
        max_drawdown_in_lockup_pct=max_dd_lockup,
        user_notes=notes,
    )
    oid = store.record_outcome(outcome)
    console.print(f"[green]✓ outcome id={oid} 已录入[/green]")
    if close:
        store.update_status(pred_id, "closed")
        console.print(f"[green]  prediction id={pred_id} 状态置为 closed[/green]")


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
    console.print(f"[bold cyan]投决归档统计[/bold cyan]")
    for k, v in s.items():
        console.print(f"  {k}: {v}")


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
