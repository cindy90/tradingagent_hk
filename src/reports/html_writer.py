"""HTML IC memo 渲染器（投行级视觉）。

设计原则：
1. 完全自包含 — 内联 CSS + 0 外部依赖, 单 .html 文件可邮件 / 打印 / 部署到任何静态托管
2. 视觉编码 — 推荐颜色（绿/黄/橙/红）/ 情景颜色 / kill_switch 严重度
3. 移动响应式 + 打印友好（@media print）
4. brief 章节用 <details> 折叠（默认收起，点击展开）
5. markdown 内容自动转 HTML（表格、列表、代码块）

参考: Anthropic 工程师 Thariq Shihipar 的 "Claude 直出 HTML 比 markdown 好用"
理念，但我们这边走"内容（markdown source of truth）+ 模板（html 视觉）"的双轨制。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.agents.base import AgentContext


_AGENT_LABELS = {
    "prospectus_analyst": "招股书深度分析",
    "industry": "行业研究",
    "macro": "宏观策略",
    "comparable": "可比公司估值",
    "tech_trend": "技术发展趋势",
    "sentiment": "二级市场情绪",
    "fact_check": "事实核对",
    "debate_manager": "Bull/Bear 辩论裁决",
    "bull": "Bull 多头观点",
    "bear": "Bear 空头观点",
    "risk": "风控独立评估",
    "decision": "最终投决",
}


def _agent_label(name: str) -> str:
    return _AGENT_LABELS.get(name, name)


def _md_to_html(text: str) -> str:
    """Markdown → HTML（表格 / 代码 / 引用 等扩展）。失败时回退到 <pre>。"""
    if not text:
        return ""
    try:
        import markdown as md
        return md.markdown(
            text,
            extensions=[
                "extra",       # tables, fenced_code, footnotes
                "sane_lists",
                "nl2br",
            ],
        )
    except ImportError:
        # markdown 包未装，简单 escape + <pre> 兜底
        from html import escape
        return f"<pre style='white-space: pre-wrap;'>{escape(text)}</pre>"


def render_ic_memo_html(ctx: AgentContext) -> str:
    """渲染 IC memo HTML 字符串。"""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as e:
        raise RuntimeError("jinja2 未安装。pip install jinja2") from e

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("ic_memo.html.j2")

    d = ctx.extras.decision_json or {}
    valuation = d.get("valuation_range_hkd_billion") or {}
    amounts = d.get("suggested_amount_usd_million") or []
    hedging = d.get("hedging_strategy")
    exit_plan = d.get("exit_plan")

    # 决议全文 markdown → html
    decision_md = ctx.full_reports.get("decision", "（决策报告缺失）")
    decision_html = _md_to_html(decision_md)

    # 各 Agent brief markdown → html (按运行顺序)
    brief_order = [
        "prospectus_analyst", "industry", "macro", "comparable",
        "tech_trend", "sentiment", "fact_check", "debate_manager",
        "risk", "decision",
    ]
    briefs_rendered: list[tuple[str, str]] = []
    for name in brief_order:
        if name in ctx.briefs:
            briefs_rendered.append((name, _md_to_html(ctx.briefs[name])))
    # 兜底: 把不在标准顺序里的 brief 也加上
    for name, brief in ctx.briefs.items():
        if name not in {n for n, _ in briefs_rendered}:
            briefs_rendered.append((name, _md_to_html(brief)))

    return template.render(
        ctx=ctx,
        d=d,
        valuation=valuation,
        amounts=amounts,
        hedging=hedging,
        exit_plan=exit_plan,
        decision_html=decision_html,
        briefs=briefs_rendered,
        agent_label=_agent_label,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def write_ic_memo_html(ctx: AgentContext, filename: str = "FINAL_MEMO.html") -> Path:
    """渲染并写入 reports_dir/<filename>.html. 返回写入的路径。"""
    html = render_ic_memo_html(ctx)
    out = ctx.reports_dir / filename
    out.write_text(html, encoding="utf-8")
    return out
