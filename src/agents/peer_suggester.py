"""可比公司候选提取器（v2: 行业池 + 业务相似度评分）。

设计 v1 → v2 改动:
- v1: 让 LLM 从招股书 RAG 自由提取 ticker → 经常空 / 编错代码
- v2: 先从 akshare 拉行业港股池（带名称+市值+主营业务），LLM 只能从池子里选 + 给业务相似度打分

工作流:
  1. workflow 启动 → 用 industry/keywords + target 估值 → build_peer_pool() 拉候选
  2. PeerSuggester 读招股书 RAG（提取业务/竞争对手描述）+ 候选池
  3. LLM (SUMMARIZE tier) 输出 [{ticker, name, similarity_score, reason}]
  4. CLI 交互式确认（按 similarity_score 排序展示）

向后兼容: pool=None 时退化到 v1 路径（让 LLM 从 RAG 自由提取，不推荐）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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

PEER_SUGGEST_SYSTEM_V2 = """你是港股 IPO 估值分析师。任务: 从给定的"行业港股候选池"中，
**结合招股书摘录**，挑出与目标公司最直接业务可比的港股 (3-8 家)。

【严格规则】
1. **只能从下方"候选池"里选**——禁止引入池外公司，即使你训练记忆里相关。
2. 池里没有合适候选时，输出空数组 []，不要凑数。
3. 业务相似度评分（similarity_score, 0-5）按以下维度综合：
   - 主营业务子类是否匹配（如协作机器人 vs 协作机器人 = 5; vs 工业机器人 = 4; vs 通用工业自动化 = 3）
   - 客户类型是否相同（B 端工业 / 服务 / 政府 / 消费）
   - 收入规模是否数量级一致（差 1 个数量级以内 = 4-5 分）
   - 商业模式是否相同（卖硬件 / SaaS / 项目制）
4. 输出严格 JSON 数组（用 ```json``` 代码块包裹），按 similarity_score **降序**:

```json
[
  {"ticker": "02432", "name": "越疆", "similarity_score": 5.0,
   "reason": "协作机器人直接竞品 + B 端工业客户 + 港股第一股"},
  {"ticker": "01021", "name": "华沿机器人", "similarity_score": 4.5,
   "reason": "工业机器人本体 + 港股 2026 年 3 月新上市 + 估值规模相近"}
]
```

【输出约束】
- ticker 必须 5 位数字字符串（不带 .HK / 空格 / 其它符号）
- name 用候选池中的公司中文名（不要自创）
- reason 1 句话 (≤50 字), 必须明确"为什么相似" + "和 target 业务的具体连接点"
- similarity_score 是 float (0-5), 一位小数

只输出一个 ```json``` 代码块, 无其他文字。"""

# 兼容老路径 (无 pool) 的 v1 prompt
PEER_SUGGEST_SYSTEM_V1 = """你是港股 IPO 估值分析师。基于招股书摘录，找出本公司的"最直接可比"已上市公司。

**严格要求**：
1. **只列港股已上市**（代码 5 位数字, .HK 后缀）。A 股 / 美股不要。
2. 公司必须是**直接业务相似**, 不是泛泛的"科技公司"。例: 协作机器人对标协作机器人, 不是泛工业自动化。
3. 如果摘录里没有港股已上市可比, 说"暂无确切可比"; **不要凭训练记忆瞎编代码**。
4. 输出严格 JSON, 仅包含一个数组, 字段 {ticker, name, similarity_score, reason}:
   - ticker: 5 位数字字符串(如 "02432" / "01021"), 不带 .HK
   - name: 公司中文简称
   - similarity_score: 0-5 浮点（默认 3.0 如不确定）
   - reason: 1 句话(<40字)说明可比度

输出格式示例:
```json
[
  {"ticker": "02432", "name": "越疆", "similarity_score": 5.0,
   "reason": "协作机器人直接竞品, 港股第一股"},
  {"ticker": "01021", "name": "华沿机器人", "similarity_score": 4.0,
   "reason": "工业机器人本体, 2026 年 3 月上市"}
]
```

只输出一个 ```json``` 代码块, 无其他文字。"""


@dataclass
class PeerCandidate:
    ticker: str
    name: str
    reason: str
    similarity_score: float = 3.0  # v2 新增, 0-5 业务相似度
    market_cap_hkd_b: float | None = None  # 来自 pool, 给 CLI 展示用


def _format_pool_for_prompt(pool: list[dict], max_show: int = 60) -> str:
    """把候选池渲染成 prompt 表格。"""
    if not pool:
        return "（候选池为空）"
    lines = ["| 代码 | 简称 | 市值（亿 HKD） | 30 日涨跌% | 主营业务（如有） |",
             "|---|---|---|---|---|"]
    for r in pool[:max_show]:
        biz = r.get("main_business") or "—"
        if len(biz) > 80:
            biz = biz[:80] + "…"
        lines.append(
            f"| {r.get('ticker','')} | {r.get('name','')} | "
            f"{r.get('market_cap_hkd_b') if r.get('market_cap_hkd_b') is not None else '—'} | "
            f"{r.get('change_pct') if r.get('change_pct') is not None else '—'} | "
            f"{biz} |"
        )
    return "\n".join(lines)


def suggest_peers(
    rag: ProspectusRAG,
    company_name: str,
    industry: str,
    llm: LLMClient,
    pool: list[dict] | None = None,
    target_market_cap_hkd_b: float | None = None,
    k: int = 8,
) -> list[PeerCandidate]:
    """从招股书 RAG + 候选池提取 peers。

    Args:
        rag: 已索引的招股书 RAG
        company_name: 标的公司中文名
        industry: 行业关键词
        llm: LLM 客户端
        pool: 候选池 (来自 build_peer_pool); None 时退化到 v1 路径
        target_market_cap_hkd_b: 目标公司估值（亿 HKD），写到 prompt 帮 LLM 判断规模匹配
        k: 每个 RAG query 召回的 chunk 数

    Returns: 候选 peer 列表（按 similarity_score 降序）; 不确定时返回空列表。
    """
    # 1. 多 query 召回招股书证据
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
    chunks = chunks[: 3 * k]

    if not chunks and not pool:
        logger.warning("[PeerSuggester] RAG 0 召回 + pool 空, 跳过 LLM")
        return []

    rag_context = "\n\n---\n\n".join(
        f"【{c.get('section','')} P.{c.get('page_start','')}-{c.get('page_end','')}】\n{c['text']}"
        for c in chunks
    ) or "（招股书 RAG 未召回相关章节）"

    # 2. v2 路径: 有 pool → 让 LLM 从池子里选
    if pool:
        pool_md = _format_pool_for_prompt(pool)
        target_size_hint = (
            f"\n# 目标公司估值规模\n约 {target_market_cap_hkd_b} 亿港元 "
            f"(选择规模相近的 peer 优先, 数量级差距 > 5x 的不要选)\n"
            if target_market_cap_hkd_b else ""
        )
        user_msg = (
            f"# 标的公司\n{company_name} (行业: {industry})\n"
            f"{target_size_hint}"
            f"\n# 行业港股候选池（你只能从这里选）\n{pool_md}\n\n"
            f"# 招股书摘录（业务/竞争描述, 用于判断相似度）\n{rag_context}\n\n"
            f"请输出最多 8 个最直接可比的 peer, 按 similarity_score 降序。"
        )
        system = PEER_SUGGEST_SYSTEM_V2
    else:
        # v1 fallback (不推荐)
        user_msg = (
            f"# 标的公司\n{company_name} (行业: {industry})\n\n"
            f"# 招股书摘录（按相关度排序）\n{rag_context}\n\n"
            f"请输出最多 5 个最直接可比的港股已上市公司。"
        )
        system = PEER_SUGGEST_SYSTEM_V1

    resp = llm.complete(
        tier=ModelTier.SUMMARIZE,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=2000,
        temperature=0.1,
    )
    candidates = _parse_candidates(resp.text)

    # 3. 用 pool 元数据补 candidate 的 market_cap, 同时校验"是否真的在池子里"
    if pool:
        pool_by_ticker = {p["ticker"]: p for p in pool}
        validated: list[PeerCandidate] = []
        for c in candidates:
            if c.ticker not in pool_by_ticker:
                logger.warning(
                    f"[PeerSuggester] LLM 输出 {c.ticker} 不在候选池中, 跳过 (防幻觉)"
                )
                continue
            c.market_cap_hkd_b = pool_by_ticker[c.ticker].get("market_cap_hkd_b")
            validated.append(c)
        candidates = validated

    # 按 similarity_score 降序
    candidates.sort(key=lambda c: c.similarity_score, reverse=True)
    return candidates


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
        m = re.search(r"\d+", ticker)
        if not m:
            continue
        ticker = m.group(0).zfill(5)
        name = str(item.get("name", "")).strip()
        reason = str(item.get("reason", "")).strip()
        # similarity_score 容错: 字符串 / int / float / 缺失
        score_raw = item.get("similarity_score", item.get("score", 3.0))
        try:
            score = float(score_raw)
            score = max(0.0, min(5.0, score))  # clamp 到 0-5
        except (TypeError, ValueError):
            score = 3.0
        if name and ticker:
            out.append(PeerCandidate(
                ticker=ticker, name=name, reason=reason,
                similarity_score=score,
            ))
    return out
