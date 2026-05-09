"""可比公司候选提取器。

在跑 9 个 agent 之前，先用招股书 RAG + 一次 LLM 调用提取候选 peers，
让用户在 CLI 里确认 / 编辑。避免后续 comparable / sentiment agent
凭 LLM 训练知识乱选可比公司。

设计：
- 不算 9 个 agent 之一, 不写报告, 不参与评分卡, 不入 ledger 主表
- 使用 SUMMARIZE tier (kimi-k2.6) 控制成本
- 输出结构化 JSON: [{ticker, name, reason}, ...]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from loguru import logger

from src.data.rag import ProspectusRAG
from src.llm import LLMClient, ModelTier

# 招股书"竞争对手"信息分布在多个章节, 用关键词召回。
RAG_QUERIES = [
    "主要竞争对手",
    "可比上市公司",
    "同业公司",
    "行业格局 市场参与者",
    "行业排名 主要厂商",
]

PEER_SUGGEST_SYSTEM = """你是港股 IPO 估值分析师。基于招股书摘录，找出本公司的"最直接可比"已上市公司。

**严格要求**：
1. **只列港股已上市**（代码 5 位数字, .HK 后缀）。A 股 / 美股不要。
2. 公司必须是**直接业务相似**, 不是泛泛的"科技公司"。例: 协作机器人对标协作机器人, 不是泛工业自动化。
3. 如果摘录里没有港股已上市可比, 说"暂无确切可比"; **不要凭训练记忆瞎编代码**。
4. 输出严格 JSON, 仅包含一个数组, 字段 {ticker, name, reason}:
   - ticker: 5 位数字字符串(如 "02432" / "01021"), 不带 .HK
   - name: 公司中文简称
   - reason: 1 句话(<40字)说明可比度

输出格式示例:
```json
[
  {"ticker": "02432", "name": "越疆", "reason": "协作机器人直接竞品, 港股第一股"},
  {"ticker": "01021", "name": "华沿机器人", "reason": "工业机器人本体, 2026 年 3 月上市"}
]
```

只输出一个 ```json``` 代码块, 无其他文字。"""


@dataclass
class PeerCandidate:
    ticker: str
    name: str
    reason: str


def suggest_peers(
    rag: ProspectusRAG,
    company_name: str,
    industry: str,
    llm: LLMClient,
    k: int = 8,
) -> list[PeerCandidate]:
    """从招股书 RAG 提取候选 peers。

    Args:
        rag: 已索引的招股书 RAG
        company_name: 标的公司中文名
        industry: 行业关键词（如 "工业机器人/协作机器人"）
        llm: LLM 客户端
        k: 每个 query 召回的 chunk 数

    Returns: 候选 peer 列表; LLM 不确定时返回空列表。
    """
    # 1. 多 query 召回
    seen_pages: set[tuple[int, int]] = set()
    chunks: list[dict] = []
    for q in RAG_QUERIES:
        for hit in rag.search(q, k=k):
            key = (hit.get("page_start", 0), hit.get("page_end", 0))
            if key in seen_pages:
                continue
            seen_pages.add(key)
            chunks.append(hit)
    chunks.sort(key=lambda h: h.get("distance") or 999)
    chunks = chunks[: 3 * k]  # 总量上限

    if not chunks:
        logger.warning("[PeerSuggester] RAG 召回 0 条, 跳过 LLM")
        return []

    context = "\n\n---\n\n".join(
        f"【{c.get('section','')} P.{c.get('page_start','')}-{c.get('page_end','')}】\n{c['text']}"
        for c in chunks
    )

    user_msg = (
        f"# 标的公司\n{company_name} (行业: {industry})\n\n"
        f"# 招股书摘录（按相关度排序）\n{context}\n\n"
        f"请输出最多 5 个最直接可比的港股已上市公司。"
    )
    resp = llm.complete(
        tier=ModelTier.SUMMARIZE,
        system=PEER_SUGGEST_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=1500,
        temperature=0.1,
    )
    return _parse_candidates(resp.text)


def _parse_candidates(text: str) -> list[PeerCandidate]:
    matches = list(re.finditer(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL))
    if not matches:
        # 兜底: 找首个 [ 到末尾 ] 的 array
        m2 = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if not m2:
            logger.warning("[PeerSuggester] LLM 输出未含 JSON 数组")
            return []
        raw = m2.group(0)
    else:
        raw = matches[-1].group(1)

    try:
        arr = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"[PeerSuggester] JSON 解析失败: {e}; raw={raw[:200]}")
        return []

    out: list[PeerCandidate] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip()
        # 标准化: 去 .HK / 空格, 保留数字
        m = re.search(r"\d+", ticker)
        if not m:
            continue
        ticker = m.group(0).zfill(5)  # 统一 5 位
        name = str(item.get("name", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if name and ticker:
            out.append(PeerCandidate(ticker=ticker, name=name, reason=reason))
    return out
