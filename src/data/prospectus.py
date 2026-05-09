"""招股书 PDF 解析与切块。

产物：ProspectusChunk 列表，每块带元数据（页码、章节标题猜测、token 估算）。
切块策略：先按页 + 标题模式切，超长页再做语义切。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger

# 港股招股书常见章节关键词（中文版优先；英文版 fallback）
SECTION_PATTERNS = [
    r"^\s*目\s*录",
    r"^\s*概\s*要",
    r"^\s*风险\s*因素",
    r"^\s*行业\s*概览",
    r"^\s*业务",
    r"^\s*财务\s*资料",
    r"^\s*管理层\s*讨论",
    r"^\s*未来\s*计划",
    r"^\s*募集\s*资金\s*用途",
    r"^\s*股本",
    r"^\s*基石\s*投资者",
    r"^\s*主要\s*股东",
    r"^\s*关连\s*交易",
    r"^\s*董事",
    # English fallback
    r"^\s*RISK\s+FACTORS",
    r"^\s*INDUSTRY\s+OVERVIEW",
    r"^\s*BUSINESS",
    r"^\s*FINANCIAL\s+INFORMATION",
    r"^\s*USE\s+OF\s+PROCEEDS",
    r"^\s*CORNERSTONE\s+INVESTORS",
]
_SECTION_RE = re.compile("|".join(f"({p})" for p in SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


@dataclass
class ProspectusChunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class ProspectusLoader:
    """加载并切块招股书。"""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

    def _extract_pages(self) -> list[tuple[int, str]]:
        """优先 pdfplumber（表格友好），失败时回退 pypdf。"""
        try:
            import pdfplumber
            pages: list[tuple[int, str]] = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, p in enumerate(pdf.pages, start=1):
                    text = p.extract_text() or ""
                    pages.append((i, text))
            return pages
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, fallback to pypdf")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(self.pdf_path))
            return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(reader.pages)]
        except Exception as e:
            logger.error(f"pypdf failed: {e}")
            return []

    @staticmethod
    def _detect_section(text: str, current: str) -> str:
        m = _SECTION_RE.search(text[:200])
        if m:
            return m.group(0).strip()
        return current

    def load_chunks(self, target_chars: int = 3000) -> list[ProspectusChunk]:
        pages = self._extract_pages()
        if not pages:
            return []

        chunks: list[ProspectusChunk] = []
        cur_section = ""
        buf_text = ""
        buf_start = pages[0][0]
        buf_end = buf_start

        def flush(buf_text: str, start: int, end: int, section: str) -> None:
            if not buf_text.strip():
                return
            chunks.append(ProspectusChunk(
                chunk_id=f"p{start}-{end}-{len(chunks)}",
                text=buf_text.strip(),
                page_start=start,
                page_end=end,
                section=section,
            ))

        for page_no, text in pages:
            cur_section = self._detect_section(text, cur_section)
            if len(buf_text) + len(text) > target_chars and buf_text:
                flush(buf_text, buf_start, buf_end, cur_section)
                buf_text = text
                buf_start = page_no
                buf_end = page_no
            else:
                buf_text += "\n" + text
                buf_end = page_no

        flush(buf_text, buf_start, buf_end, cur_section)
        logger.info(f"招股书切块完成: {len(pages)} 页 → {len(chunks)} chunks")
        return chunks
