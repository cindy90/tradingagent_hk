"""招股书 PDF 解析与切块。

产物：ProspectusChunk 列表，每块带元数据（页码、章节标题猜测、token 估算、类型）。

切块策略 (#16):
- token-aware 切块（不再用 char count）：BGE-zh-v1.5 max input 512 token，
  字符数和 token 数对中英混排差异巨大，按字符切会被 embedding 静默截断
- 滑动窗口 + overlap，提升跨 chunk RAG 召回（关键概念可能跨 chunk 边界）
- 章节边界优先切：检测到新章节强制 flush，避免跨章节混编

表格抽取 (#17):
- pdfplumber.page.extract_tables() 抽取表格 → markdown 表格 → 独立 chunk
  metadata={"type": "table"}
- 财务三表、可比公司估值表、募资明细表都是结构化数据，
  原 extract_text() 会拍平为长字符串，RAG 检索"营收增速"基本失效
- 抽出来后表格独立索引，"营收 增速 CAGR" 类查询能直接命中财务表
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

# BGE-zh max input 512 token，留 112 余量
DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50


def estimate_bge_tokens(text: str) -> int:
    """估算 BGE-zh (BertTokenizer) 的 token 数。

    经验值（适用于中英混排的港股招股书）:
      - 汉字 1:1 → token
      - ASCII 字母词 1.3 token / word
      - 数字/标点平均 0.4 token / char

    实测在 BGE-zh-v1.5 上误差 ±10%，足够用于切块决策。
    比起加载 transformers tokenizer 这种估算无依赖、零成本。
    """
    if not text:
        return 0
    chinese = sum(1 for c in text if "一" <= c <= "鿿")
    en_words = len(re.findall(r"[A-Za-z]+", text))
    other = len(text) - chinese - sum(len(w) for w in re.findall(r"[A-Za-z]+", text))
    return chinese + int(en_words * 1.3) + max(0, other // 3)


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


# ---------- 表格转 markdown ----------

def _table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 抽出的二维表格转 markdown 表格。"""
    rows = [["" if c is None else str(c).strip().replace("\n", " ") for c in row]
            for row in table if row]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for r in body:
        # 行长度对齐
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[:len(header)]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _is_meaningful_table(table: list[list]) -> bool:
    """过滤明显的伪表格（页眉、单元格全空、行数<2）。"""
    if not table or len(table) < 2:
        return False
    non_empty = sum(1 for row in table for c in row if c and str(c).strip())
    return non_empty >= 4


# ---------- 主 Loader ----------

class ProspectusLoader:
    """加载并切块招股书。"""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

    def _extract_pages(self) -> list[tuple[int, str, list[list[list]]]]:
        """返回 [(page_no, text, tables), ...]。

        优先 pdfplumber（同时拿到 text + tables），失败回退 pypdf（无 tables）。
        """
        try:
            import pdfplumber
            out: list[tuple[int, str, list[list[list]]]] = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, p in enumerate(pdf.pages, start=1):
                    text = p.extract_text() or ""
                    try:
                        tables = p.extract_tables() or []
                    except Exception:
                        tables = []
                    out.append((i, text, tables))
            return out
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, fallback to pypdf (无表格)")

        try:
            from pypdf import PdfReader
            reader = PdfReader(str(self.pdf_path))
            return [(i + 1, (pg.extract_text() or ""), [])
                    for i, pg in enumerate(reader.pages)]
        except Exception as e:
            logger.error(f"pypdf failed: {e}")
            return []

    @staticmethod
    def _detect_section(text: str, current: str) -> str:
        m = _SECTION_RE.search(text[:200])
        if m:
            return m.group(0).strip()
        return current

    @staticmethod
    def _split_text_token_aware(
        text: str,
        target_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """把单段长文本按 token 估算切成多段，带 overlap。

        以段落（双换行）/ 句号为优先切点；找不到时按字符硬切。
        """
        if not text.strip():
            return []
        if estimate_bge_tokens(text) <= target_tokens:
            return [text]

        # 按段落 / 句号切成最小单元
        units = re.split(r"(\n\s*\n|[。！？\.\!\?]\s+)", text)
        # 重组：相邻 unit + delimiter 拼一起
        atoms: list[str] = []
        i = 0
        while i < len(units):
            atom = units[i]
            if i + 1 < len(units):
                atom += units[i + 1]
            atoms.append(atom)
            i += 2

        chunks: list[str] = []
        cur: list[str] = []
        cur_tok = 0
        for atom in atoms:
            atom_tok = estimate_bge_tokens(atom)
            if cur_tok + atom_tok > target_tokens and cur:
                chunks.append("".join(cur))
                # overlap: 保留尾部 overlap_tokens 的 atom 作为下一 chunk 起头
                tail: list[str] = []
                tail_tok = 0
                for a in reversed(cur):
                    a_tok = estimate_bge_tokens(a)
                    if tail_tok + a_tok > overlap_tokens:
                        break
                    tail.insert(0, a)
                    tail_tok += a_tok
                cur = tail + [atom]
                cur_tok = tail_tok + atom_tok
            else:
                cur.append(atom)
                cur_tok += atom_tok
        if cur:
            chunks.append("".join(cur))

        # 极端情况：单 atom 已超 target，硬切
        final: list[str] = []
        for c in chunks:
            if estimate_bge_tokens(c) <= target_tokens * 1.2:
                final.append(c)
                continue
            # 按字符硬切
            char_per_token = max(1, len(c) // max(estimate_bge_tokens(c), 1))
            target_chars = target_tokens * char_per_token
            for j in range(0, len(c), target_chars):
                final.append(c[j:j + target_chars])
        return [c for c in final if c.strip()]

    def load_chunks(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        extract_tables: bool = True,
    ) -> list[ProspectusChunk]:
        pages = self._extract_pages()
        if not pages:
            return []

        chunks: list[ProspectusChunk] = []
        cur_section = ""
        n_text = 0
        n_table = 0

        for page_no, text, tables in pages:
            cur_section = self._detect_section(text, cur_section)

            # 文本切块（token-aware）
            if text.strip():
                pieces = self._split_text_token_aware(text, target_tokens, overlap_tokens)
                for j, piece in enumerate(pieces):
                    chunks.append(ProspectusChunk(
                        chunk_id=f"text-p{page_no}-{j}",
                        text=piece.strip(),
                        page_start=page_no,
                        page_end=page_no,
                        section=cur_section,
                        metadata={"type": "text", "tokens": estimate_bge_tokens(piece)},
                    ))
                    n_text += 1

            # 表格独立成 chunk
            if extract_tables and tables:
                for k, tbl in enumerate(tables):
                    if not _is_meaningful_table(tbl):
                        continue
                    md = _table_to_markdown(tbl)
                    if not md or estimate_bge_tokens(md) > target_tokens * 3:
                        # 表格过大时按行切，保留表头
                        header_line = md.split("\n")[0] if md else ""
                        sep_line = md.split("\n")[1] if md and "\n" in md else ""
                        body_lines = md.split("\n")[2:] if md.count("\n") >= 2 else []
                        cur_lines: list[str] = []
                        cur_tok = estimate_bge_tokens(header_line + sep_line)
                        sub = 0
                        for ln in body_lines:
                            ln_tok = estimate_bge_tokens(ln)
                            if cur_tok + ln_tok > target_tokens and cur_lines:
                                tbl_text = "\n".join([header_line, sep_line] + cur_lines)
                                chunks.append(ProspectusChunk(
                                    chunk_id=f"table-p{page_no}-{k}-{sub}",
                                    text=tbl_text,
                                    page_start=page_no,
                                    page_end=page_no,
                                    section=cur_section,
                                    metadata={"type": "table",
                                              "tokens": estimate_bge_tokens(tbl_text)},
                                ))
                                n_table += 1
                                sub += 1
                                cur_lines = [ln]
                                cur_tok = estimate_bge_tokens(header_line + sep_line) + ln_tok
                            else:
                                cur_lines.append(ln)
                                cur_tok += ln_tok
                        if cur_lines:
                            tbl_text = "\n".join([header_line, sep_line] + cur_lines)
                            chunks.append(ProspectusChunk(
                                chunk_id=f"table-p{page_no}-{k}-{sub}",
                                text=tbl_text,
                                page_start=page_no,
                                page_end=page_no,
                                section=cur_section,
                                metadata={"type": "table",
                                          "tokens": estimate_bge_tokens(tbl_text)},
                            ))
                            n_table += 1
                    else:
                        chunks.append(ProspectusChunk(
                            chunk_id=f"table-p{page_no}-{k}",
                            text=md,
                            page_start=page_no,
                            page_end=page_no,
                            section=cur_section,
                            metadata={"type": "table",
                                      "tokens": estimate_bge_tokens(md)},
                        ))
                        n_table += 1

        logger.info(
            f"招股书切块完成: {len(pages)} 页 → "
            f"{n_text} 文本块 + {n_table} 表格块 = {len(chunks)} chunks "
            f"(target={target_tokens} tok, overlap={overlap_tokens} tok)"
        )
        return chunks
