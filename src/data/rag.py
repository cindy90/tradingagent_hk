"""招股书 RAG 索引。每个 IPO 项目独立 collection，避免跨项目污染。

Embedding 默认使用 BGE-zh-v1.5（专为中文检索优化），通过 sentence-transformers 加载。
首次使用会自动下载模型权重到 ~/.cache/huggingface。配置通过 EMBEDDING_MODEL 覆盖。

BGE v1.5 在 query 端建议加 prefix 以提升召回；本实现在 search() 中自动加 prefix，
而 index() 端保持原文。这样既符合 BGE 推荐用法，又不影响其它非 BGE 模型。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from loguru import logger

from config import get_settings

from .prospectus import ProspectusChunk

# BGE-zh-v1.5 推荐的 query 前缀
_BGE_QUERY_PREFIX_ZH = "为这个句子生成表示以用于检索相关文章："


class ProspectusRAG:
    def __init__(self, project_id: str, embedding_model: str | None = None):
        self.project_id = project_id
        s = get_settings()
        self.embedding_model_name = embedding_model or s.embedding_model
        self.embedding_device = s.embedding_device
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _build_embedding_fn(self):
        try:
            from chromadb.utils import embedding_functions  # type: ignore
        except ImportError as e:
            raise RuntimeError("chromadb 未安装") from e

        # SentenceTransformerEmbeddingFunction 内部用 sentence-transformers
        # 直接接收 HF 模型 ID（如 BAAI/bge-base-zh-v1.5）
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=self.embedding_device,
            normalize_embeddings=True,
        )

    def _ensure(self):
        if self._collection is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise RuntimeError("chromadb 未安装，请 pip install chromadb") from e

        s = get_settings()
        persist_dir = s.cache_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = self._build_embedding_fn()
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # collection 名带模型 hash 后缀，避免切换 embedding 模型时检索维度冲突
        coll_name = f"prospectus_{self.project_id}_{self._safe_model_tag()}"
        self._collection = self._client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model_name},
            embedding_function=self._embedding_fn,
        )

    def _safe_model_tag(self) -> str:
        return self.embedding_model_name.split("/")[-1].replace(".", "_").replace("-", "_")[:32]

    def _is_bge_zh(self) -> bool:
        m = self.embedding_model_name.lower()
        return "bge" in m and ("zh" in m or "chinese" in m)

    def index(self, chunks: Iterable[ProspectusChunk]) -> int:
        self._ensure()
        chunks = list(chunks)
        if not chunks:
            return 0
        self._collection.upsert(  # type: ignore[union-attr]
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"section": c.section, "page_start": c.page_start, "page_end": c.page_end}
                for c in chunks
            ],
        )
        logger.info(
            f"RAG indexed {len(chunks)} chunks (model={self.embedding_model_name}) "
            f"for {self.project_id}"
        )
        return len(chunks)

    def search(self, query: str, k: int = 6) -> list[dict]:
        self._ensure()
        q = (_BGE_QUERY_PREFIX_ZH + query) if self._is_bge_zh() else query
        res = self._collection.query(query_texts=[q], n_results=k)  # type: ignore[union-attr]
        out = []
        for i, doc in enumerate(res["documents"][0]):
            out.append({
                "text": doc,
                "section": res["metadatas"][0][i].get("section", ""),
                "page_start": res["metadatas"][0][i].get("page_start", 0),
                "page_end": res["metadatas"][0][i].get("page_end", 0),
                "distance": res["distances"][0][i] if "distances" in res else None,
            })
        return out

    def is_indexed(self) -> bool:
        self._ensure()
        return self._collection.count() > 0  # type: ignore[union-attr]
