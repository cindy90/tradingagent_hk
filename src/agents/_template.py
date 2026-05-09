"""通用 Agent 模板。子类只需声明 name/system_prompt/user_prompt_builder 即可。

把所有"读上下文 → 调 LLM → 摘要 → 落盘"的样板代码统一在这里。新增 Agent
只需要写 prompt，专注于专业内容本身。
"""
from __future__ import annotations

from typing import Callable

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.agents.summarizer import Summarizer
from src.llm import ModelTier


class TemplateAgent(BaseAgent):
    """模板 Agent。子类必须覆盖 name/description/SYSTEM/build_user_message。"""
    SYSTEM: str = ""

    def __init__(self, llm, summarizer: Summarizer | None = None):
        super().__init__(llm)
        self.summarizer = summarizer or Summarizer(llm)

    def build_user_message(self, ctx: AgentContext) -> str:
        raise NotImplementedError

    def run(self, ctx: AgentContext) -> AgentReport:
        user_msg = self.build_user_message(ctx)
        resp = self.llm.complete(
            tier=self.tier,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            cached_system_blocks=ctx.cached_blocks or None,
            max_tokens=4000,
            temperature=0.3,
        )
        full = resp.text
        brief = self.summarizer.compress(full, agent_name=self.name)
        ctx.full_reports[self.name] = full
        ctx.briefs[self.name] = brief
        return AgentReport(agent=self.name, full_report=full, brief=brief)


def briefs_context(ctx: AgentContext, agents: list[str]) -> str:
    """从 ctx 中拼接指定 agents 的 brief，用于辩论 / 风控 / 决策类 Agent。"""
    parts = []
    for name in agents:
        if name in ctx.briefs:
            parts.append(f"### {name}\n{ctx.briefs[name]}")
    return "\n\n".join(parts) if parts else "（暂无简报）"
