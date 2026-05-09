"""中间层摘要器。把任意 Agent 的全文报告压缩成 <500 字的结构化 brief。

用 Haiku 跑，单次成本极低（一份 5000 字报告 → 输入 ~3K tokens / 输出 ~500 tokens
≈ $0.005）。这是 token 优化的关键节点：保证下游 Agent 不读全文。
"""
from __future__ import annotations

from src.llm import LLMClient, ModelTier

SUMMARIZER_SYSTEM = """你是一名资深投行分析师助理，专门为投决会准备简报。

你的任务：把输入的完整研究报告压缩为一份结构化简报，要求：
1. 严格控制在 500 字以内（中文计字符数）
2. 保留关键数字、关键结论、关键风险，删除推理过程
3. 用以下固定结构输出（Markdown）：

```
**核心结论**：<1-2 句话>
**关键事实**：
- <事实 1，必须包含数字>
- <事实 2>
- <事实 3-5>
**支持点**：<1-2 句>
**风险点**：<1-2 句>
**置信度**：<高/中/低> + 一句理由
```
不要输出任何额外文字。"""


class Summarizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compress(self, full_report: str, agent_name: str = "") -> str:
        prompt = f"待压缩的研究报告（来自 {agent_name}）：\n\n{full_report}"
        resp = self.llm.complete(
            tier=ModelTier.SUMMARIZE,
            system=SUMMARIZER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,
        )
        return resp.text.strip()
