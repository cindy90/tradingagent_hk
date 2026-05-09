"""CLI 入口。

用法:
    python cli.py analyze --ticker 09999 --name "示例科技" --industry "AI/SaaS" --pdf data/prospectus/09999.pdf

可选参数:
    --debate-rounds N        覆盖默认辩论轮数
    --no-prospectus          跳过招股书加载（快速测试）
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from config import get_settings
from src.data.hkex_client import download_document
from src.data.ths_client import THSClient
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
    decision = ctx.extras.get("decision_json")
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


@app.command()
def show_config() -> None:
    """打印当前配置。"""
    s = get_settings()
    console.print(s.model_dump())


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
