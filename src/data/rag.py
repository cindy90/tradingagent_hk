"""招股书 RAG 索引。每个 IPO 项目独立 collection，避免跨项目污染。

embedding 模型默认用 chromadb 内置的 sentence-transformers all-MiniLM-L6-v2，
中文效果一般但够用；如需更好可换 BGE-zh 或 OpenAI/Voyage embeddings。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from loguru import logger

from config import get_settings

from .prospectus import ProspectusChunk


class ProspectusRAG:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self._client = None
        self._collection = None

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

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=f"prospectus_{self.project_id}",
            metadata={"hnsw:space": "cosine"},
        )

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
        logger.info(f"RAG indexed {len(chunks)} chunks for {self.project_id}")
        return len(chunks)

    def search(self, query: str, k: int = 6) -> list[dict]:
        self._ensure()
        res = self._collection.query(query_texts=[query], n_results=k)  # type: ignore[union-attr]
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
