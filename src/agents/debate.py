"""Bull/Bear 辩论协调器 + 辩论裁判（Manager）。

辩论流程：
- Round 0: Bull 出第一稿 → Bear 出第一稿（互不可见对方观点）
- Round 1..N: Bull 看 Bear 后修订 → Bear 看 Bull 后修订
- Manager 收敛：基于最后一轮双方观点，输出辩论结论摘要

辩论收敛策略：每轮结束后比较 Bull/Bear brief 的 token 重合度（jaccard），
高于阈值或达到最大轮数则停止。这是一个简单但有效的 token 节省策略。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.agents.base import AgentContext, AgentReport, BaseAgent
from src.agents.bear import BearResearcher
from src.agents.bull import BullResearcher
from src.agents.summarizer import Summarizer
from src.llm import LLMClient, ModelTier


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


MANAGER_SYSTEM = """你是投决会主席，负责裁决 Bull / Bear 辩论。基于双方的最新论点，输出：

## 一、Bull 主要论据（提炼 3-5 条）
## 二、Bear 主要论据（提炼 3-5 条）
## 三、双方共识点
## 四、核心分歧点（明确指出哪 1-3 个问题决定胜负）
## 五、裁决倾向（Bull / Bear / 中性）+ 理由
## 六、关键不确定性（需在风控环节进一步评估的事项）

输出 800-1200 字。

【关键约束 — 严禁幻觉】
- 你的工作是**裁决** Bull / Bear 已提出的论据, 不是引入新信息。
- 不得引入 Bull / Bear 未提到的新数字、新公司、新事件作为裁决依据。
- 共识点 / 分歧点必须基于双方文本里都提到（共识）或一方提到另一方未否认（分歧）的内容。
- 裁决倾向必须给出**概率化**的理由（如"Bear 框架仅需单点失效, Bull 需 4 假设同时成立, 概率乘积低"），
  不要写"Bear 更稳"等情绪化表述。
- 关键不确定性必须**可量化追踪**（如"Q1 现金流 < 阈值即否决"）, 不要写"市场情绪变化"等模糊条件。"""


@dataclass
class DebateOutcome:
    rounds: int
    bull_history: list[str]
    bear_history: list[str]
    manager_summary: str


class DebateOrchestrator(BaseAgent):
    name = "debate_manager"
    tier = ModelTier.DECIDE
    description = "Bull/Bear 辩论裁判"

    def __init__(self, llm: LLMClient, max_rounds: int = 2, convergence_threshold: float = 0.7):
        super().__init__(llm)
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.summarizer = Summarizer(llm)
        self.bull = BullResearcher(llm, self.summarizer)
        self.bear = BearResearcher(llm, self.summarizer)

    def run(self, ctx: AgentContext) -> AgentReport:
        bull_history: list[str] = []
        bear_history: list[str] = []

        for r in range(self.max_rounds + 1):
            logger.info(f"[Debate] Round {r}")
            bull_rep = self.bull.run(ctx)
            bear_rep = self.bear.run(ctx)
            bull_history.append(bull_rep.brief)
            bear_history.append(bear_rep.brief)

            sim = _jaccard(bull_rep.brief, bear_rep.brief)
            logger.debug(f"[Debate] R{r} jaccard={sim:.2f}")
            if sim >= self.convergence_threshold and r >= 1:
                logger.info(f"[Debate] 早停于 R{r}（相似度 {sim:.2f}）")
                break

        user_msg = (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})\n\n"
            f"# Bull 最终观点\n{bull_history[-1]}\n\n"
            f"# Bear 最终观点\n{bear_history[-1]}\n\n"
            f"请输出辩论裁决。"
        )
        resp = self.llm.complete(
            tier=self.tier,
            system=MANAGER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2500,
            temperature=0.2,
        )
        manager_full = resp.text
        manager_brief = self.summarizer.compress(manager_full, agent_name=self.name)

        ctx.full_reports[self.name] = manager_full
        ctx.briefs[self.name] = manager_brief
        ctx.extras.debate_outcome = DebateOutcome(
            rounds=len(bull_history),
            bull_history=bull_history,
            bear_history=bear_history,
            manager_summary=manager_brief,
        )

        return AgentReport(
            agent=self.name,
            full_report=manager_full,
            brief=manager_brief,
            metadata={"rounds": len(bull_history)},
        )
