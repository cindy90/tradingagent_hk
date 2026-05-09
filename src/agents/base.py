"""Agent 基类。

设计要点:
1. 每个 Agent 输出 (full_report, brief) 两份:
   - full_report: 完整 markdown，落盘到 reports/<project>/<step>_<agent>.md，供用户审查；
   - brief: <500 字结构化摘要 JSON / 短文本，作为下游 Agent 的输入。
   下游 Agent 默认只读 brief，避免长上下文叠加。
2. 通过 LLMClient 的 cached_system_blocks 复用招股书等大段静态内容。
3. 工具调用走 RAG/akshare/确定性计算，不让 LLM 推数。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.extras import WorkflowExtras
from src.data.rag import ProspectusRAG
from src.llm import LLMClient, ModelTier


@dataclass
class AgentContext:
    """跨 Agent 共享的运行上下文。"""
    project_id: str
    ticker: str
    company_name: str
    industry: str
    reports_dir: Path
    rag: ProspectusRAG | None = None
    briefs: dict[str, str] = field(default_factory=dict)
    full_reports: dict[str, str] = field(default_factory=dict)
    extras: WorkflowExtras = field(default_factory=WorkflowExtras)
    cached_blocks: list[str] = field(default_factory=list)


@dataclass
class AgentReport:
    agent: str
    full_report: str
    brief: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"
    tier: ModelTier = ModelTier.ANALYZE
    description: str = ""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentReport:
        ...

    def _save_full_report(self, ctx: AgentContext, step_no: int, body: str) -> Path:
        path = ctx.reports_dir / f"{step_no:02d}_{self.name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {self.description or self.name}\n\n"
            f"- 项目: {ctx.company_name} ({ctx.ticker})\n"
            f"- Agent: `{self.name}` (tier={self.tier.value})\n"
            f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n---\n\n"
        )
        path.write_text(header + body, encoding="utf-8")
        logger.info(f"[{self.name}] 报告已写入 {path}")
        return path
