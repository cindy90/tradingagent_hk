"""通用 Agent 模板。子类只需声明 name/system_prompt/user_prompt_builder 即可。

把所有"读上下文 → 调 LLM → 摘要 → 落盘"的样板代码统一在这里。新增 Agent
只需要写 prompt，专注于专业内容本身。

ScoreCard 集成 (Phase A):
- 子类可选定义 score_card_class，会自动:
  * 在 SYSTEM 末尾追加 schema 指令，要求 LLM 在报告后产出 ```json``` 评分卡
  * 解析 LLM 输出中的 json 块，落到 ctx.extras.misc["score_cards"][agent_name]
  * 报告 markdown 自动剥离 json 块（人看的版本）
"""
from __future__ import annotations

from typing import Callable

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.agents.scoring import parse_score_card, schema_instruction, strip_score_card_block
from src.agents.summarizer import Summarizer
from src.feedback.models import AgentScoreCard
from src.llm import ModelTier


class TemplateAgent(BaseAgent):
    """模板 Agent。子类必须覆盖 name/description/SYSTEM/build_user_message。"""
    SYSTEM: str = ""
    score_card_class: type[AgentScoreCard] | None = None

    def __init__(self, llm, summarizer: Summarizer | None = None):
        super().__init__(llm)
        self.summarizer = summarizer or Summarizer(llm)

    def build_user_message(self, ctx: AgentContext) -> str:
        raise NotImplementedError

    def _effective_system(self) -> str:
        """SYSTEM + （可选）评分卡 schema 指令。"""
        if self.score_card_class is None:
            return self.SYSTEM
        return self.SYSTEM + schema_instruction(self.score_card_class)

    def run(self, ctx: AgentContext) -> AgentReport:
        user_msg = self.build_user_message(ctx)
        resp = self.llm.complete(
            tier=self.tier,
            system=self._effective_system(),
            messages=[{"role": "user", "content": user_msg}],
            cached_system_blocks=ctx.cached_blocks or None,
            max_tokens=4500,
            temperature=0.3,
        )
        raw = resp.text

        # 评分卡解析（不阻塞主流程，失败时记录 warning）
        if self.score_card_class is not None:
            sc = parse_score_card(raw, self.score_card_class, agent_name=self.name)
            if sc is not None:
                cards = ctx.extras.misc.setdefault("score_cards", {})
                cards[self.name] = sc.model_dump()

        # markdown 报告剥离 json 块
        full = strip_score_card_block(raw) if self.score_card_class else raw
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
