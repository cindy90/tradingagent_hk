"""最终汇总报告生成器：把所有 Agent 的简报 + 决策 JSON 拼成一份给投决会的 Final Memo。

格式参考顶级投行 IC memo: 顶部 1 页 Executive Summary, 后接敏感性分析、对冲策略、
kill switches、详细 KPI 表, 最后是各 Agent 简报合集。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.base import AgentContext


def _render_executive_summary(d: dict) -> str:
    """顶部 1 页 Executive Summary（IC memo 标志）。"""
    rec = d.get("recommendation", "—")
    conf = d.get("confidence", "—")
    pricing = d.get("ipo_pricing_view", "—")
    val = d.get("valuation_range_hkd_billion") or {}
    amt = d.get("suggested_amount_usd_million") or [None, None]
    val_low = val.get("low", "—")
    val_mid = val.get("mid", "—")
    val_high = val.get("high", "—")
    anchor = val.get("anchor_method", "—")

    return (
        f"## ⭐ Executive Summary\n\n"
        f"| 字段 | 值 |\n"
        f"|---|---|\n"
        f"| **建议** | **{rec}** |\n"
        f"| 置信度 | {conf} |\n"
        f"| 定价观点 | {pricing} |\n"
        f"| 估值区间（亿 HKD） | {val_low} / {val_mid} / {val_high}（low / mid / high） |\n"
        f"| 估值锚定方法 | {anchor} |\n"
        f"| 建议认购金额（USD M） | {amt[0]} - {amt[1] if len(amt) > 1 else amt[0]} |\n"
    )


def _render_sensitivity_table(scenarios: list) -> str:
    if not scenarios:
        return "_（敏感性分析缺失，建议手工补全或重跑 decision）_"
    lines = [
        "| 情景 | 触发条件 | 估值（亿 HKD） | 概率 | 预期回报 |",
        "|---|---|---|---|---|",
    ]
    for s in scenarios:
        triggers = "; ".join(s.get("triggers", []) if isinstance(s, dict) else getattr(s, "triggers", []))
        get = (lambda k: s.get(k, "—")) if isinstance(s, dict) else (lambda k: getattr(s, k, "—"))
        prob = get("probability")
        ret = get("expected_return_pct")
        lines.append(
            f"| **{get('name')}** | {triggers[:120]} | {get('valuation_hkd_b')} | "
            f"{prob*100 if isinstance(prob, (int, float)) else prob}% | "
            f"{f'{ret:+.1f}%' if isinstance(ret, (int, float)) else (ret or '—')} |"
        )
    return "\n".join(lines)


def _render_kill_switches(switches: list) -> str:
    if not switches:
        return "_（未提供 kill switches）_"
    lines = ["| # | 触发事件 | 动作 | 严重度 |", "|---|---|---|---|"]
    for i, s in enumerate(switches, 1):
        get = (lambda k: s.get(k, "—")) if isinstance(s, dict) else (lambda k: getattr(s, k, "—"))
        lines.append(f"| {i} | {get('trigger')} | {get('action')} | {get('severity')} |")
    return "\n".join(lines)


def _render_hedging(h) -> str:
    if not h:
        return "_（未提供对冲策略）_"
    get = (lambda k: h.get(k, "—")) if isinstance(h, dict) else (lambda k: getattr(h, k, "—"))
    cov = get("target_coverage_pct")
    cov_str = f"{cov*100:.0f}%" if isinstance(cov, (int, float)) else (cov or "—")
    return (
        f"- **工具**: {get('instrument')}\n"
        f"- **覆盖比例（名义本金占比）**: {cov_str}\n"
        f"- **理由**: {get('rationale')}\n"
    )


def _render_exit_plan(e) -> str:
    if not e:
        return "_（未提供退出策略）_"
    get = (lambda k: e.get(k, "—")) if isinstance(e, dict) else (lambda k: getattr(e, k, "—"))
    triggers = get("trigger_conditions")
    if isinstance(triggers, list) and triggers:
        triggers_md = "\n  - " + "\n  - ".join(triggers)
    else:
        triggers_md = " 无"
    return (
        f"- **退出时点**: {get('horizon')}\n"
        f"- **方式**: {get('method')}\n"
        f"- **节奏**: {get('pace')}\n"
        f"- **提前减持触发条件**:{triggers_md}\n"
    )


def _render_monitoring_kpis(detailed: list, fallback: list) -> str:
    if detailed:
        lines = ["| KPI | 阈值 | 频率 | 违阈动作 |", "|---|---|---|---|"]
        for k in detailed:
            get = (lambda x: k.get(x, "—")) if isinstance(k, dict) else (lambda x: getattr(k, x, "—"))
            lines.append(f"| {get('name')} | {get('threshold')} | {get('frequency')} | {get('action_if_breach')} |")
        return "\n".join(lines)
    if fallback:
        return "\n".join(f"- {x}" for x in fallback)
    return "_（未提供监控 KPI）_"


def _render_list(items: list, prefix: str = "- ") -> str:
    if not items:
        return "_（无）_"
    return "\n".join(f"{prefix}{i}" for i in items)


def write_final_summary(ctx: AgentContext, *, also_html: bool = True) -> Path:
    """生成 FINAL_MEMO.md 主文件。

    Args:
        also_html: True (默认) 时同时输出 FINAL_MEMO.html (投行级视觉, 邮件友好)。
                   失败时静默回退（不阻塞 markdown 输出）。
    """
    d = ctx.extras.decision_json or {}
    decision_json_str = json.dumps(d, ensure_ascii=False, indent=2) if d else "{}"

    # Executive Summary
    exec_md = _render_executive_summary(d)

    # 敏感性分析
    sensitivity_md = _render_sensitivity_table(d.get("sensitivity_table") or [])

    # 对冲 / 退出 / kill switches
    hedging_md = _render_hedging(d.get("hedging_strategy"))
    exit_md = _render_exit_plan(d.get("exit_plan"))
    kill_md = _render_kill_switches(d.get("kill_switches") or [])

    # 监控 KPI
    monitor_md = _render_monitoring_kpis(
        d.get("monitoring_kpis_detailed") or [],
        d.get("monitoring_kpis") or [],
    )

    # 关键支持 / 风险 / 条件
    supports_md = _render_list(d.get("key_supports") or [])
    risks_md = _render_list(d.get("key_risks") or [])
    conditions_md = _render_list(d.get("deal_conditions") or [])

    # 各 Agent brief
    brief_section = "\n\n".join(
        f"### `{name}`\n\n{brief}" for name, brief in ctx.briefs.items()
    )

    md = f"""# 基石投资决策备忘录 (IC Memo)

- 项目：**{ctx.company_name}** ({ctx.ticker})
- 行业：{ctx.industry}
- 项目 ID：`{ctx.project_id}`

---

{exec_md}

---

## 一、关键支持论点

{supports_md}

## 二、核心风险

{risks_md}

## 三、敏感性分析（三档情景）⭐

{sensitivity_md}

## 四、对冲策略 ⭐

{hedging_md}

## 五、退出策略 ⭐

{exit_md}

## 六、Kill Switches（认购后退出触发）⭐

{kill_md}

## 七、监控 KPI

{monitor_md}

## 八、硬性认购条件 (Deal Conditions)

{conditions_md}

---

## 九、决议全文

{ctx.full_reports.get("decision", "（决策报告缺失）")}

---

## 十、各环节简报合集

{brief_section}

---

## 附录：完整决议 JSON

```json
{decision_json_str}
```

> 详细分项报告见同目录下 `01_*.md` 至 `10_*.md`。
"""
    out = ctx.reports_dir / "FINAL_MEMO.md"
    out.write_text(md, encoding="utf-8")

    if also_html:
        try:
            from src.reports.html_writer import write_ic_memo_html
            html_path = write_ic_memo_html(ctx)
            from loguru import logger
            logger.info(f"FINAL_MEMO.html 已生成: {html_path}")
        except Exception as e:
            from loguru import logger
            logger.warning(f"HTML 渲染失败（不影响 markdown 输出）: {e}")

    return out
