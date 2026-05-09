"""跨项目案例 RAG。把 (Prediction + Outcome + Postmortem) 三元组索引到独立 collection，
新项目分析时按行业/规模/估值维度检索 top-k 相似案例，注入 prompt 作为经验。

设计:
- 单 collection "ipo_cases"（不像招股书 RAG 每项目一份）
- 文档 = 案例摘要（结构化 markdown ~600-1000 token / case）
- 元数据: ticker / industry / valuation_mid / recommendation / d180_return / status
- 查询时按 industry + 估值规模匹配，结合 embedding 相似度

为什么不全文索引？
- 文档少（一年几十单），全文索引边际收益低、token 高
- 案例摘要的"教训提炼"段落是真正有价值的迁移信号
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import get_settings
from src.feedback.models import Outcome, Prediction, Score


_BGE_QUERY_PREFIX_ZH = "为这个句子生成表示以用于检索相关文章："


def render_case_summary(p: Prediction, o: Outcome | None, s: Score | None) -> str:
    """把一个案例渲染成给 RAG 检索 / 给 LLM 阅读的结构化摘要。"""
    lines = [
        f"# 案例: {p.ticker} {p.company_name}",
        f"- 决策日期: {p.decision_date:%Y-%m-%d}",
        f"- 行业: {p.industry}",
        f"- 估值中枢: {p.valuation_mid} 亿港元 (区间 {p.valuation_low}-{p.valuation_high}, {p.anchor_method})",
        f"- 我们的建议: **{p.recommendation}** (置信度: {p.confidence})",
        f"- 定价观点: {p.ipo_pricing_view}",
        "",
        f"## 当时的关键支持",
    ]
    lines += [f"- {x}" for x in p.key_supports[:5]]
    lines += ["", f"## 当时识别的风险"]
    lines += [f"- {x}" for x in p.key_risks[:5]]

    if o is not None:
        lines += ["", f"## 实际投后表现"]
        if o.ipo_actual_marketcap_hkd_billion is not None:
            lines.append(f"- 实际市值: {o.ipo_actual_marketcap_hkd_billion} 亿港元")
        if o.d1_return is not None:
            lines.append(f"- D1: {o.d1_return * 100:.1f}%")
        if o.d180_return is not None:
            lines.append(f"- D180 (禁售期满): {o.d180_return * 100:.1f}%")
        if o.d365_return is not None:
            lines.append(f"- D365: {o.d365_return * 100:.1f}%")
        if o.was_broken_ipo_d180 is not None:
            lines.append(f"- 6 月内破发: {'是' if o.was_broken_ipo_d180 else '否'}")
        if o.cornerstone_realized_return_pct is not None:
            lines.append(f"- 解禁日实际收益: {o.cornerstone_realized_return_pct * 100:.1f}%")
        if o.notable_events:
            lines.append(f"- 重大事件: {', '.join(o.notable_events)}")

    if s is not None and s.postmortem_memo:
        # 只取教训章节（节省 token，最具迁移价值）
        memo = s.postmortem_memo
        idx = memo.find("## 四、教训")
        lessons = memo[idx:idx + 1500] if idx >= 0 else memo[-1500:]
        lines += ["", f"## 复盘教训", lessons.strip()]
        if s.error_root_causes:
            lines.append(f"\n根因分类: {', '.join(s.error_root_causes)}")
        if s.recommendation_score is not None:
            lines.append(f"决议事后正确度评分: {s.recommendation_score:.2f}  (-1=完全错 1=完全对)")

    return "\n".join(lines)


class CaseRAG:
    COLLECTION_NAME = "ipo_cases"

    def __init__(self, embedding_model: str | None = None):
        s = get_settings()
        self.embedding_model_name = embedding_model or s.embedding_model
        self.embedding_device = s.embedding_device
        self._client = None
        self._collection = None

    def _ensure(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise RuntimeError("chromadb 未安装") from e

        s = get_settings()
        persist_dir = s.cache_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=self.embedding_device,
            normalize_embeddings=True,
        )
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        coll_tag = self.embedding_model_name.split("/")[-1].replace("-", "_").replace(".", "_")[:32]
        self._collection = self._client.get_or_create_collection(
            name=f"{self.COLLECTION_NAME}_{coll_tag}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=emb_fn,
        )

    def _is_bge_zh(self) -> bool:
        m = self.embedding_model_name.lower()
        return "bge" in m and ("zh" in m or "chinese" in m)

    def index_case(
        self,
        prediction: Prediction,
        outcome: Outcome | None,
        score: Score | None,
    ) -> None:
        self._ensure()
        text = render_case_summary(prediction, outcome, score)
        meta = {
            "project_id": prediction.project_id,
            "ticker": prediction.ticker,
            "industry": prediction.industry or "",
            "valuation_mid": float(prediction.valuation_mid or 0),
            "recommendation": prediction.recommendation or "",
            "decision_date": prediction.decision_date.isoformat() if prediction.decision_date else "",
            "d180_return": float(outcome.d180_return) if outcome and outcome.d180_return is not None else 0.0,
            "has_outcome": outcome is not None,
            "has_score": score is not None,
        }
        self._collection.upsert(  # type: ignore[union-attr]
            ids=[prediction.project_id],
            documents=[text],
            metadatas=[meta],
        )

    def search(
        self,
        industry: str,
        valuation_mid: float | None = None,
        recommendation_hint: str | None = None,
        k: int = 3,
        only_with_outcome: bool = True,
    ) -> list[dict]:
        """检索相似历史案例。

        策略：embedding 相似 + metadata 过滤（仅保留已有 outcome 的案例，
        没有 outcome 的案例还没"答案"，参考价值低）。
        """
        self._ensure()
        if self._collection.count() == 0:  # type: ignore[union-attr]
            return []

        # 构造查询文本（让 embedding 兼顾行业/估值/推荐）
        parts = [f"行业 {industry}"]
        if valuation_mid:
            scale = self._scale_label(valuation_mid)
            parts.append(f"估值 {scale}")
        if recommendation_hint:
            parts.append(f"建议类型 {recommendation_hint}")
        query = "  ".join(parts)
        if self._is_bge_zh():
            query = _BGE_QUERY_PREFIX_ZH + query

        where: dict[str, Any] = {}
        if only_with_outcome:
            where["has_outcome"] = True

        kwargs: dict = {"query_texts": [query], "n_results": k * 2}  # 多取一些后过滤
        if where:
            kwargs["where"] = where
        try:
            res = self._collection.query(**kwargs)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(f"CaseRAG 检索失败: {e}")
            return []

        hits = []
        for i, doc in enumerate(res["documents"][0]):
            md = res["metadatas"][0][i]
            hits.append({
                "text": doc,
                "metadata": md,
                "distance": res["distances"][0][i] if "distances" in res else None,
            })
        # 估值规模相近的排前面（粗过滤后再按 embedding 距离）
        if valuation_mid:
            hits.sort(key=lambda h: (
                abs(h["metadata"].get("valuation_mid", 0) - valuation_mid) / max(valuation_mid, 1),
                h.get("distance", 0),
            ))
        return hits[:k]

    @staticmethod
    def _scale_label(v: float) -> str:
        if v < 30:
            return "小型 30 亿港元以下"
        if v < 100:
            return "中型 30-100 亿港元"
        if v < 300:
            return "大型 100-300 亿港元"
        return "超大型 300 亿港元以上"

    def count(self) -> int:
        self._ensure()
        return self._collection.count()  # type: ignore[union-attr]

    @staticmethod
    def format_for_prompt(hits: list[dict]) -> str:
        """把检索结果渲染成注入 system prompt 的字符串。"""
        if not hits:
            return ""
        lines = [
            "# 历史相似案例参考（来自系统过往决策的复盘）",
            "",
            "下列是与本次项目类似的历史决策及其实际投后表现，**特别关注其中的"
            "教训章节**，避免重复同类错误：",
            "",
        ]
        for i, h in enumerate(hits, 1):
            md = h["metadata"]
            lines.append(f"## 案例 {i}: {md.get('ticker')} ({md.get('industry')})")
            lines.append(h["text"][:2500])
            lines.append("")
        return "\n".join(lines)


def reindex_all_cases(store: "FeedbackStore | None" = None) -> int:
    """从 SQLite 全量重建 CaseRAG 索引。日常无需调用，仅在 embedding 模型切换或库损坏时用。"""
    from src.feedback.store import FeedbackStore as _Store

    store = store or _Store()
    rag = CaseRAG()
    n = 0
    for p in store.list_predictions(limit=10000):
        pred_row = store._conn.execute(
            "SELECT id FROM predictions WHERE project_id=?", (p.project_id,)
        ).fetchone()
        pid = pred_row["id"]
        o = store.latest_outcome(pid)
        s = store.latest_score(pid)
        rag.index_case(p, o, s)
        n += 1
    logger.info(f"CaseRAG 重建完成: {n} 个案例")
    return n
