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
    debate_rounds: int | None = typer.Option(None, "--debate-rounds", help="辩论最大轮数"),
    no_prospectus: bool = typer.Option(False, "--no-prospectus", help="跳过招股书"),
) -> None:
    """对单个 IPO 项目跑完整深度分析。"""
    _setup_logging()
    s = get_settings()
    if not s.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY 未设置，请先配置 .env[/red]")
        raise typer.Exit(code=1)

    pdf_arg = None if no_prospectus else (pdf or _try_default_pdf(ticker))
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


if __name__ == "__main__":
    app()
