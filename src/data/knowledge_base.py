"""T5: KnowledgeBaseRAG — 通用本地知识库 RAG (与 ProspectusRAG 并列).

定位: ProspectusRAG 索引**单个招股书** (项目级 collection); KnowledgeBaseRAG
索引**用户的通用知识库** (跨项目共享): 监管笔记 / 行业研究 / 历史复盘 /
投决纪要 / 个人 take 等。

为什么不接 IMA / NotebookLM 直连?
- 两者都没可用 public API; IMA 完全封闭, NotebookLM 仅企业版有受限 API
  且国区不可用
- 当外挂会绑死你换工具的自由
- 走"导出 → 本地文件 → ingest"链路, 你可以自由切换笔记工具
  (Obsidian / Notion 导出 / IMA 导出 / 任意 markdown / PDF)

支持的文件格式 (自动按扩展名识别):
- .md / .markdown — 按 H1/H2/H3 切块, 保留 heading 作 metadata
- .txt — 按段落 (\\n\\n) 切, 大段做 1k char 滑窗
- .pdf — 用 pdfplumber 抽文本, 按段切

幂等性: 每个文件算 SHA-256 hash; 重跑 ingest 同文件未变直接跳过, 变了
就 delete 旧 chunks + 重插.

引用格式: 检索结果含 source_path + heading/page, agents 可生成
  [KB: 监管笔记/2025-10 SFC 18C 收紧.md#"非传统行业准入"]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from config import get_settings


SUPPORTED_EXTS = (".md", ".markdown", ".txt", ".pdf")


# ============================================================================
# 切块 (loader)
# ============================================================================

@dataclass
class KBChunk:
    """KB 单个 chunk."""
    chunk_id: str               # 全局唯一: <file_hash[:8]>_<idx>
    text: str
    source_path: str            # 相对 ingest_root 的路径
    source_type: str            # md / txt / pdf
    heading: str = ""           # md 的 H1>H2>H3 串联; pdf 无意义留空
    page: int = 0               # pdf 的页码; md/txt 留 0
    chunk_idx: int = 0
    file_hash: str = ""


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def _split_paragraphs(text: str, *, max_chars: int = 1000) -> list[str]:
    """段落切 + 大段滑窗 (1000 chars / 200 overlap)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            out.append(p)
            continue
        # 滑窗
        i, step = 0, max_chars - 200
        while i < len(p):
            out.append(p[i: i + max_chars])
            i += step
    return out


def load_markdown(
    path: Path, source_path: str, file_hash: str,
) -> list[KBChunk]:
    """按 heading 切. 每个 heading 下的 body 还按段落 max_chars 切."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    # 用 ATX heading (# / ## / ###) 切; 维护 heading stack
    sections: list[tuple[str, list[str]]] = []  # (heading_str, body_lines)
    stack: list[str] = []
    current_body: list[str] = []
    current_head = ""

    def flush():
        if current_body or current_head:
            sections.append((current_head, current_body[:]))

    for ln in lines:
        m = re.match(r"^(#{1,3})\s+(.+?)\s*$", ln)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2)
            # 截断 stack 到 level-1, 然后 push
            stack = stack[: level - 1] + [title]
            current_head = " > ".join(stack)
            current_body = []
        else:
            current_body.append(ln)
    flush()
    if not sections:
        sections = [("", lines)]

    out: list[KBChunk] = []
    idx = 0
    for head, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for piece in _split_paragraphs(body):
            out.append(KBChunk(
                chunk_id=f"{file_hash[:8]}_{idx}",
                text=piece, source_path=source_path,
                source_type="md", heading=head, page=0,
                chunk_idx=idx, file_hash=file_hash,
            ))
            idx += 1
    return out


def load_text(
    path: Path, source_path: str, file_hash: str,
) -> list[KBChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: list[KBChunk] = []
    for idx, piece in enumerate(_split_paragraphs(text)):
        out.append(KBChunk(
            chunk_id=f"{file_hash[:8]}_{idx}", text=piece,
            source_path=source_path, source_type="txt",
            heading="", page=0, chunk_idx=idx, file_hash=file_hash,
        ))
    return out


def load_pdf(
    path: Path, source_path: str, file_hash: str,
) -> list[KBChunk]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning(f"[KB] pdfplumber 未安装, 跳过 PDF: {source_path}")
        return []
    out: list[KBChunk] = []
    idx = 0
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                if not txt.strip():
                    continue
                for piece in _split_paragraphs(txt):
                    out.append(KBChunk(
                        chunk_id=f"{file_hash[:8]}_{idx}", text=piece,
                        source_path=source_path, source_type="pdf",
                        heading="", page=page_no,
                        chunk_idx=idx, file_hash=file_hash,
                    ))
                    idx += 1
    except Exception as e:
        logger.warning(f"[KB] PDF 解析失败 {source_path}: {e}")
        return []
    return out


def load_file(
    path: Path, *, ingest_root: Path,
) -> tuple[str, list[KBChunk]]:
    """根据扩展名 dispatch loader; 返回 (file_hash, chunks)."""
    ext = path.suffix.lower()
    rel = str(path.relative_to(ingest_root))
    h = _file_hash(path)
    if ext in (".md", ".markdown"):
        return h, load_markdown(path, rel, h)
    if ext == ".txt":
        return h, load_text(path, rel, h)
    if ext == ".pdf":
        return h, load_pdf(path, rel, h)
    return h, []


# ============================================================================
# 索引 (ChromaDB)
# ============================================================================

class KnowledgeBaseRAG:
    """通用知识库 RAG. 与 ProspectusRAG 共用 ChromaDB 持久化目录,
    但是独立 collection (跨项目共享)."""

    COLLECTION_PREFIX = "knowledge_base"

    def __init__(self, embedding_model: str | None = None):
        s = get_settings()
        self.embedding_model_name = embedding_model or s.embedding_model
        self.embedding_device = s.embedding_device
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _safe_model_tag(self) -> str:
        m = self.embedding_model_name.split("/")[-1]
        return re.sub(r"[^a-zA-Z0-9_]", "_", m)[:32]

    def _is_bge_zh(self) -> bool:
        m = self.embedding_model_name.lower()
        return "bge" in m and ("zh" in m or "chinese" in m)

    def _build_embedding_fn(self):
        from chromadb.utils import embedding_functions  # type: ignore
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=self.embedding_device,
            normalize_embeddings=True,
        )

    def _ensure(self):
        if self._collection is not None:
            return
        import chromadb  # type: ignore
        from chromadb.config import Settings as ChromaSettings  # type: ignore

        s = get_settings()
        persist_dir = s.cache_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = self._build_embedding_fn()
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        coll_name = f"{self.COLLECTION_PREFIX}_{self._safe_model_tag()}"
        self._collection = self._client.get_or_create_collection(
            name=coll_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model_name,
            },
            embedding_function=self._embedding_fn,
        )

    # ---------- 文件级幂等控制 ----------

    def _file_hash_lookup(self, source_path: str) -> str | None:
        """查 collection 里这个 source_path 的现存 file_hash."""
        self._ensure()
        try:
            res = self._collection.get(  # type: ignore[union-attr]
                where={"source_path": source_path}, limit=1,
            )
        except Exception:
            return None
        metas = res.get("metadatas") or []
        if not metas:
            return None
        return (metas[0] or {}).get("file_hash")

    def _delete_file(self, source_path: str) -> int:
        """删 source_path 下所有 chunk; 返删除数."""
        self._ensure()
        try:
            res = self._collection.get(  # type: ignore[union-attr]
                where={"source_path": source_path},
            )
        except Exception:
            return 0
        ids = res.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)  # type: ignore[union-attr]
        return len(ids)

    # ---------- 主流程 ----------

    def index_chunks(self, chunks: Iterable[KBChunk]) -> int:
        """直接落 chunks (upsert)."""
        self._ensure()
        chunks = list(chunks)
        if not chunks:
            return 0
        self._collection.upsert(  # type: ignore[union-attr]
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_path": c.source_path,
                    "source_type": c.source_type,
                    "heading": c.heading or "",
                    "page": c.page,
                    "chunk_idx": c.chunk_idx,
                    "file_hash": c.file_hash,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def ingest_directory(
        self, root: Path | str, *, recursive: bool = True,
    ) -> dict[str, int]:
        """递归扫目录 → 按扩展名 load → 幂等 upsert.

        Returns: 统计 dict {
          'files_scanned', 'files_skipped_unchanged',
          'files_skipped_unsupported', 'files_indexed',
          'chunks_inserted', 'chunks_deleted_stale',
        }
        """
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"KB 目录不存在: {root}")

        files = (
            [p for p in root.rglob("*") if p.is_file()] if recursive
            else [p for p in root.iterdir() if p.is_file()]
        )

        stats = {
            "files_scanned": 0,
            "files_skipped_unchanged": 0,
            "files_skipped_unsupported": 0,
            "files_indexed": 0,
            "chunks_inserted": 0,
            "chunks_deleted_stale": 0,
        }
        for p in files:
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTS:
                stats["files_skipped_unsupported"] += 1
                continue
            stats["files_scanned"] += 1
            rel = str(p.relative_to(root))
            new_hash = _file_hash(p)
            old_hash = self._file_hash_lookup(rel)
            if old_hash == new_hash:
                stats["files_skipped_unchanged"] += 1
                continue
            if old_hash:
                stats["chunks_deleted_stale"] += self._delete_file(rel)
            _, chunks = load_file(p, ingest_root=root)
            if chunks:
                stats["chunks_inserted"] += self.index_chunks(chunks)
                stats["files_indexed"] += 1
        logger.info(f"[KB] ingest {root}: {stats}")
        return stats

    # ---------- 检索 ----------

    _BGE_QUERY_PREFIX_ZH = "为这个句子生成表示以用于检索相关文章："

    def search(
        self, query: str, *, k: int = 5,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索. 返回每个命中的 text + source_path + heading/page + distance.

        source_type: 'md' / 'txt' / 'pdf' 仅过滤特定类型; None 不过滤.
        """
        self._ensure()
        q = (self._BGE_QUERY_PREFIX_ZH + query) if self._is_bge_zh() else query
        kwargs: dict[str, Any] = {"query_texts": [q], "n_results": k}
        if source_type:
            kwargs["where"] = {"source_type": source_type}
        res = self._collection.query(**kwargs)  # type: ignore[union-attr]
        out = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] if "distances" in res else []
        for i, doc in enumerate(docs):
            md = metas[i] if i < len(metas) else {}
            out.append({
                "text": doc,
                "source_path": md.get("source_path", ""),
                "source_type": md.get("source_type", ""),
                "heading": md.get("heading", ""),
                "page": md.get("page", 0),
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def count(self) -> int:
        self._ensure()
        return self._collection.count()  # type: ignore[union-attr]

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """列出所有已索引文件 (source_path + file_hash + chunk count)."""
        self._ensure()
        res = self._collection.get()  # type: ignore[union-attr]
        metas = res.get("metadatas") or []
        agg: dict[str, dict[str, Any]] = {}
        for md in metas:
            if not md:
                continue
            sp = md.get("source_path", "")
            entry = agg.setdefault(sp, {
                "source_path": sp,
                "source_type": md.get("source_type", ""),
                "file_hash": md.get("file_hash", ""),
                "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(agg.values(), key=lambda r: r["source_path"])


# ============================================================================
# Agent 集成 helper (render markdown brief)
# ============================================================================

def render_kb_results_md(
    results: list[dict[str, Any]], *, max_chars_per_hit: int = 400,
) -> str:
    """把 KB 检索结果渲染成 markdown, 喂 agent prompt 用.

    引用格式: [KB: <source_path>#<heading>] 或 [KB: <source_path>:p<N>]
    """
    if not results:
        return "(知识库无相关内容)"
    lines = ["### 知识库检索结果", ""]
    for i, r in enumerate(results, 1):
        sp = r.get("source_path", "?")
        head = r.get("heading", "")
        page = r.get("page", 0)
        if head:
            ref = f"[KB: {sp}#{head}]"
        elif page:
            ref = f"[KB: {sp}:p{page}]"
        else:
            ref = f"[KB: {sp}]"
        txt = (r.get("text") or "").strip()
        if len(txt) > max_chars_per_hit:
            txt = txt[:max_chars_per_hit] + "..."
        dist = r.get("distance")
        dist_str = f" (dist={dist:.3f})" if isinstance(dist, (int, float)) else ""
        lines.append(f"**[{i}]** {ref}{dist_str}\n\n{txt}\n")
    return "\n".join(lines)


def query_knowledge_base(
    query: str, *, k: int = 5, source_type: str | None = None,
) -> list[dict[str, Any]]:
    """便捷封装: agent 直接调一次, 不用自己 init."""
    return KnowledgeBaseRAG().search(query, k=k, source_type=source_type)


# ============================================================================
# CLI
# ============================================================================

def _cmd_ingest(args) -> int:
    kb = KnowledgeBaseRAG()
    stats = kb.ingest_directory(Path(args.dir).expanduser(), recursive=not args.no_recursive)
    print(f"\n=== KB ingest done ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n总 chunk 数 (collection): {kb.count()}")
    return 0


def _cmd_query(args) -> int:
    kb = KnowledgeBaseRAG()
    if kb.count() == 0:
        print("[ERROR] KB 为空, 先跑 ingest <dir>.", file=sys.stderr)
        return 2
    hits = kb.search(args.query, k=args.k, source_type=args.type)
    print(render_kb_results_md(hits))
    return 0


def _cmd_list(args) -> int:
    kb = KnowledgeBaseRAG()
    files = kb.list_indexed_files()
    if not files:
        print("(KB 为空)")
        return 0
    print(f"已索引 {len(files)} 个文件:")
    for f in files:
        print(
            f"  {f['source_type']:<5} {f['chunks']:>4} chunks  "
            f"{f['file_hash'][:8]}  {f['source_path']}"
        )
    print(f"\n总 chunk 数: {kb.count()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="knowledge_base",
        description=(
            "通用本地知识库 RAG. 索引 markdown / pdf / txt 目录, "
            "供 agents 查询. 与 ProspectusRAG 独立, 跨项目共享."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="递归索引目录")
    p_ing.add_argument("dir", help="目录绝对路径")
    p_ing.add_argument("--no-recursive", action="store_true")
    p_ing.set_defaults(func=_cmd_ingest)

    p_q = sub.add_parser("query", help="语义检索")
    p_q.add_argument("query", help="检索语句")
    p_q.add_argument("-k", type=int, default=5)
    p_q.add_argument("--type", choices=("md", "pdf", "txt"), default=None)
    p_q.set_defaults(func=_cmd_query)

    p_l = sub.add_parser("list", help="列已索引文件")
    p_l.set_defaults(func=_cmd_list)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
