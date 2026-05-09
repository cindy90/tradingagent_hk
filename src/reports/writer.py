"""最终汇总报告生成器：把所有 Agent 的简报 + 决策 JSON 拼成一份给投决会的 Final Memo。"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.base import AgentContext


def write_final_summary(ctx: AgentContext) -> Path:
    decision_json = ctx.extras.decision_json or {}
    decision_json_str = json.dumps(decision_json, ensure_ascii=False, indent=2) if decision_json else "{}"

    brief_section = "\n\n".join(
        f"### {name}\n{brief}" for name, brief in ctx.briefs.items()
    )

    md = f"""# 基石投资决策备忘录

- 项目：{ctx.company_name} ({ctx.ticker})
- 行业：{ctx.industry}
- 项目 ID：`{ctx.project_id}`

---

## 一、最终决议（结构化）

```json
{decision_json_str}
```

## 二、决议全文

{ctx.full_reports.get("decision", "（决策报告缺失）")}

## 三、各环节简报合集

{brief_section}

---

> 详细分项报告见同目录下 `01_*.md` 至 `09_*.md`。
"""
    out = ctx.reports_dir / "FINAL_MEMO.md"
    out.write_text(md, encoding="utf-8")
    return out
