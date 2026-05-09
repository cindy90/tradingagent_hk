"""ListingProfileDetector — 从招股书 RAG 自动推断 ListingProfile。

工作流程:
1. 多 query RAG 召回包含上市规则关键词的段落
2. 一次 LLM 调用 (SUMMARIZE tier, 便宜) 输出结构化 JSON
3. 用 ListingProfile.model_validate 校验, 失败时返空字典让上层用默认

只检测 LLM 能从招股书可靠推出的字段:
- listing_chapter (18A/18C/19C/标准/二次/AH 等)
- profitability_stage (盈利状态)
- has_wvr (同股不同权)
- has_a_share_listed (是否已 A 股上市) + a_share_ticker
- is_concept_stock (中概股)
- main_listing_market (二次上市的主上市地)
- industry_theme (从招股书业务描述推断)
- detection_evidence (招股书页码)

注意: size_tier 不依赖招股书, 由 CLI 显式或 target_market_cap 推算, 这里不检测。
"""
from __future__ import annotations

import json
import re

from loguru import logger

from src.data.rag import ProspectusRAG
from src.llm import LLMClient, ModelTier


# 关键词召回 query (覆盖招股书会出现"上市规则章节标识"的章节)
_RAG_QUERIES = [
    # 18A
    "未盈利生物科技 第十八A章 18A 章 Chapter 18A",
    # 18C
    "特专科技 第十八C章 18C 章 Specialist Technology Chapter 18C",
    # 19C / WVR
    "不同投票权 同股不同权 双重股权 WVR weighted voting rights 第八A章",
    # AH 双重
    "已于上海证券交易所 已于深圳证券交易所 A 股上市 已发行 A 股",
    # 二次上市
    "二次上市 Secondary Listing 主要上市地 已于纳斯达克 已于纽约证券交易所",
    # 盈利状态
    "盈利测试 盈利能力测试 财务资格 净利润 收入 营业收入 累计亏损",
    # 业务描述（行业主题）
    "主营业务 业务概览 公司概况",
]


SYSTEM_PROMPT = """你是港股上市规则与公司画像识别专家。任务: 从招股书摘录中**严格按证据**推断:
- listing_chapter (HKEX 上市规则章节)
- profitability_stage (盈利阶段)
- industry_theme (行业主题大类)
- has_wvr / has_a_share_listed / a_share_ticker / is_concept_stock / main_listing_market

【硬约束】
1. 每个字段只能从招股书摘录中**有明确文字证据**时才填具体值; 否则填 "Unknown" / null / false。
2. **禁止凭训练记忆填**——例如不能因为公司名是"XX 生物" 就推 listing_chapter=Main_Board_18A,
   必须找到"根据上市规则第 18A 章""未盈利生物科技公司"等明确字句。
3. 输出 JSON 字段值必须使用以下枚举:
   - listing_chapter: Main_Board_Standard / Main_Board_18A / Main_Board_18C /
                      Main_Board_19C / Secondary_Listing / Dual_Primary_AH / GEM / Unknown
   - profitability_stage: Profitable_Stable / Profitable_Growth / Pre_Profit_Late_Stage /
                          Loss_Making_Growth / Pre_Commercial / Unknown
   - industry_theme: Tech_AI_Semi / Bio_Pharma / Med_Device / Robotics_Automation /
                     New_Energy / Advanced_Materials / Consumer / Financial /
                     Real_Estate / Industrial / Healthcare / Other
4. detection_evidence: 列出每个非 Unknown 字段的招股书页码 + 关键句子摘抄
   (例: "P.45: 'The Company is a specialist technology company under Chapter 18C'")

输出严格 JSON (用 ```json``` 代码块包裹):

```json
{
  "listing_chapter": "...",
  "profitability_stage": "...",
  "industry_theme": "...",
  "has_wvr": false,
  "has_a_share_listed": false,
  "a_share_ticker": "",
  "is_concept_stock": false,
  "main_listing_market": "",
  "detection_evidence": [
    "P.X: '原文摘录...' → 推断 listing_chapter=...",
    "P.Y: '原文摘录...' → 推断 has_wvr=true"
  ]
}
```

只输出一段 ```json``` 代码块, 不要其他文字。"""


def detect_listing_profile(
    rag: ProspectusRAG,
    company_name: str,
    llm: LLMClient,
    *,
    k_per_query: int = 4,
    max_chunks: int = 25,
) -> dict | None:
    """从招股书 RAG 自动推断 ListingProfile 的字段。

    Returns: dict (可直接传给 ListingProfile.model_validate)。失败时返 None。
    """
    seen: set[tuple[int, int]] = set()
    chunks: list[dict] = []
    for q in _RAG_QUERIES:
        try:
            hits = rag.search(q, k=k_per_query)
        except Exception as e:
            logger.warning(f"[Detector] RAG 检索失败 query={q!r}: {e}")
            continue
        for h in hits:
            key = (h.get("page_start", 0), h.get("page_end", 0))
            if key in seen:
                continue
            seen.add(key)
            chunks.append(h)
    if not chunks:
        logger.info("[Detector] RAG 召回 0 条, 无法推断")
        return None

    chunks.sort(key=lambda h: h.get("distance") or 999)
    chunks = chunks[:max_chunks]

    context = "\n\n---\n\n".join(
        f"【{c.get('section', '')} P.{c.get('page_start', '')}-{c.get('page_end', '')}】\n"
        f"{c.get('text', '')[:1200]}"
        for c in chunks
    )

    user_msg = (
        f"# 标的公司\n{company_name}\n\n"
        f"# 招股书摘录（按相关度排序）\n{context}\n\n"
        f"请输出严格的 ListingProfile JSON。"
    )

    try:
        resp = llm.complete(
            tier=ModelTier.SUMMARIZE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1500,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning(f"[Detector] LLM 调用失败: {e}")
        return None

    return _parse_profile_json(resp.text)


def _parse_profile_json(text: str) -> dict | None:
    matches = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    if matches:
        raw = matches[-1].group(1)
    else:
        m2 = re.search(r"\{[^{}]*\"listing_chapter\".*?\}", text, re.DOTALL)
        if not m2:
            logger.warning("[Detector] LLM 输出未含 JSON")
            return None
        raw = m2.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"[Detector] JSON 解析失败: {e}")
        return None
    if not isinstance(data, dict):
        return None
    # 清洗字段
    cleaned: dict = {}
    for k in (
        "listing_chapter", "profitability_stage", "industry_theme",
        "a_share_ticker", "main_listing_market",
    ):
        v = data.get(k)
        if isinstance(v, str) and v:
            cleaned[k] = v
    for k in ("has_wvr", "has_a_share_listed", "is_concept_stock"):
        v = data.get(k)
        if isinstance(v, bool):
            cleaned[k] = v
    evidence = data.get("detection_evidence")
    if isinstance(evidence, list):
        cleaned["detection_evidence"] = [str(x) for x in evidence][:10]
    return cleaned
