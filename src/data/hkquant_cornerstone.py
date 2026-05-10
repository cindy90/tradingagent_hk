"""hkquant cornerstone 投资人解析 + 历史业绩查询 (T2).

本文件**部分代码 vendoring 自** cindy90/hkquant 项目的 src/data/dao.py:
- _LEGAL_SUFFIXES / _PUNCT_RE / normalize_cs_name
- _tokens / _jaccard
- resolve_cornerstone_id (5 策略模糊匹配)

复制 (而非 pip 依赖) 是为了:
1. 解耦 — 他们改 dao.py 不影响本项目运行
2. 上游函数签名是 (conn, ...), 我们封装为 (raw_name, *, db_path) 更方便

源参考: https://github.com/cindy90/hkquant/blob/main/src/data/dao.py

新增 (本项目自有):
- get_cornerstone_master(cornerstone_id) → 投资人画像
- get_cornerstone_performance_asof(cornerstone_id, asof) → 业绩快照
- enrich_cornerstone_names(raw_names, *, asof) → 主流程 API
  对一组招股书原文中的基石名称, 返回每个的解析结果 + 历史业绩
"""
from __future__ import annotations

import re as _re
import sqlite3
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher as _SeqMatcher
from pathlib import Path
from typing import Any, Optional, Tuple

from loguru import logger

from src.data.hkquant_client import _connect, is_available


# ============================================================================
# Vendored: alias 归一化 + 5 策略模糊匹配
# (源: cindy90/hkquant src/data/dao.py — 修改时请同步上游)
# ============================================================================

_LEGAL_SUFFIXES = (
    "有限合伙企业", "私募基金管理有限公司", "私募基金管理", "投资管理有限公司",
    "国际投资有限公司", "资产管理有限公司", "投资有限公司", "管理有限公司",
    "有限责任公司", "股份有限公司", "股份合作公司", "有限公司", "(集团)",
    "（集团）", "private limited", "asset management", "asia pacific",
    "international", "investment", "investments", "capital", "holdings",
    "company", "limited", "incorporated", "corporation", "co.,ltd",
    "co.,ltd.", "co., ltd", "co. ltd", "co. ltd.", "ltd.", "ltd",
    "inc.", "inc", "llc", "l.p.", "lp", "spv", "spc", "plc",
    "(spc)", "(spv)", "(hk)", "(香港)", "（香港）", "香港",
)
_PUNCT_RE = _re.compile(r"[()（）.,。， '\"\-/\\&’‘]+")


def normalize_cs_name(name: str) -> str:
    """归一化基石名称用于模糊匹配 (vendored from hkquant.dao).

    步骤: 小写 → 去括号注释 → 剥常见法人后缀 → 标点转空格 → 折叠空白
    """
    s = name.strip().lower()
    if not s:
        return ""
    s = _re.sub(r"\([^)]*\)", " ", s)
    s = _re.sub(r"（[^）]*）", " ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed:
        changed = False
        for suf in _LEGAL_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)].rstrip()
                changed = True
                break
    return s


def _tokens(s: str) -> set:
    out: set = set()
    for word in s.split():
        if _re.search(r"[a-z0-9]", word):
            out.add(word)
        else:
            for ch in word:
                if ch.strip():
                    out.add(ch)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def resolve_cornerstone_id(
    conn: sqlite3.Connection,
    raw_name: str,
    *,
    min_confidence: float = 0.40,
) -> Optional[Tuple[str, float]]:
    """5 策略模糊匹配 (vendored from hkquant.dao).

    返回 (cornerstone_id, confidence); confidence < min_confidence 视为未命中.
    """
    raw_lower = raw_name.strip().lower()
    if not raw_lower:
        return None

    # 策略 1: 精确匹配
    row = conn.execute(
        "SELECT cornerstone_id, match_confidence FROM cornerstone_aliases "
        "WHERE alias_text_lower = ? ORDER BY match_confidence DESC LIMIT 1",
        (raw_lower,),
    ).fetchone()
    if row:
        return row["cornerstone_id"], row["match_confidence"]

    all_rows = conn.execute(
        "SELECT cornerstone_id, alias_text_lower, match_confidence "
        "FROM cornerstone_aliases"
    ).fetchall()
    if not all_rows:
        return None

    raw_norm = normalize_cs_name(raw_name)
    raw_tokens = _tokens(raw_norm) if raw_norm else _tokens(raw_lower)

    # 策略 2: 归一化精确
    if raw_norm:
        for r in all_rows:
            if normalize_cs_name(r["alias_text_lower"]) == raw_norm:
                conf = r["match_confidence"] * 0.95
                if conf >= min_confidence:
                    return r["cornerstone_id"], conf

    # 策略 3: 子串包含
    sub_hits = [
        r for r in all_rows
        if r["alias_text_lower"] and (
            r["alias_text_lower"] in raw_lower
            or raw_lower in r["alias_text_lower"]
        )
    ]
    if sub_hits:
        best = max(
            sub_hits,
            key=lambda r: len(r["alias_text_lower"]) * r["match_confidence"],
        )
        conf = best["match_confidence"] * 0.70
        if conf >= min_confidence:
            return best["cornerstone_id"], conf

    # 策略 4: Token Jaccard
    best_j: Tuple[Optional[str], float] = (None, 0.0)
    for r in all_rows:
        alias_norm = normalize_cs_name(r["alias_text_lower"])
        alias_tokens = (
            _tokens(alias_norm) if alias_norm
            else _tokens(r["alias_text_lower"])
        )
        j = _jaccard(raw_tokens, alias_tokens)
        score = j * r["match_confidence"] * 0.85
        if score > best_j[1]:
            best_j = (r["cornerstone_id"], score)
    if best_j[0] and best_j[1] >= min_confidence:
        return best_j[0], best_j[1]

    # 策略 5: SequenceMatcher
    best_s: Tuple[Optional[str], float] = (None, 0.0)
    for r in all_rows:
        target = normalize_cs_name(r["alias_text_lower"]) or r["alias_text_lower"]
        ratio = _SeqMatcher(None, raw_norm or raw_lower, target).ratio()
        score = ratio * r["match_confidence"] * 0.70
        if score > best_s[1]:
            best_s = (r["cornerstone_id"], score)
    if best_s[0] and best_s[1] >= min_confidence:
        return best_s[0], best_s[1]

    return None


# ============================================================================
# 本项目自有: master / performance / 主流程 API
# ============================================================================

@dataclass
class CornerstoneInfo:
    """单个基石投资人在 hkquant 中的画像 + 历史业绩."""
    raw_name: str                 # 招股书原文
    cornerstone_id: str | None    # 解析出的 ID; None = 未命中
    confidence: float             # 0-1
    canonical_name: str | None
    cornerstone_type: str | None  # PE / Sovereign / Family / Strategic 等
    country_of_origin: str | None
    aum_usd_latest: float | None
    is_chinese: bool
    is_longterm: bool             # hkquant 标注的"长线投资人"
    # performance_asof
    ipo_count_5y: int | None
    avg_m6_return_5y: float | None    # 0-1 小数 (hkquant 原格式)
    winrate_m6_5y: float | None       # 0-1
    avg_d30_return_5y: float | None   # 0-1
    lockup_discipline_score: float | None  # 0-1
    sector_expertise: str | None      # GICS L2 字符串


def _row_to_master(row: sqlite3.Row | None) -> dict[str, Any]:
    if not row:
        return {}
    return dict(row)


def get_cornerstone_master(
    cornerstone_id: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """读 cornerstone_master 单行; 未命中返 {}."""
    if not is_available() and db_path is None:
        return {}
    sql = "SELECT * FROM cornerstone_master WHERE cornerstone_id = ?"
    try:
        with _connect(db_path) as conn:
            return _row_to_master(conn.execute(sql, (cornerstone_id,)).fetchone())
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant.cs] get_cornerstone_master 失败: {e}")
        return {}


def get_cornerstone_performance_asof(
    cornerstone_id: str,
    *,
    asof: date | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """读 cornerstone_performance_asof, 取 ≤ asof 最新一行; 未命中返 {}."""
    if not is_available() and db_path is None:
        return {}
    asof_str = asof.isoformat() if isinstance(asof, date) else (
        str(asof) if asof else date.today().isoformat()
    )
    sql = (
        "SELECT * FROM cornerstone_performance_asof "
        "WHERE cornerstone_id = ? AND as_of_date <= ? "
        "ORDER BY as_of_date DESC LIMIT 1"
    )
    try:
        with _connect(db_path) as conn:
            row = conn.execute(sql, (cornerstone_id, asof_str)).fetchone()
            return dict(row) if row else {}
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant.cs] get_cornerstone_performance_asof 失败: {e}")
        return {}


def enrich_cornerstone_names(
    raw_names: list[str],
    *,
    asof: date | str | None = None,
    min_confidence: float = 0.40,
    db_path: Path | str | None = None,
) -> list[CornerstoneInfo]:
    """主流程: 一组招股书原文基石名称 → 每个的解析 ID + 画像 + 业绩.

    Args:
        raw_names: 招股书提取的基石原文 (可能是 "高瓴资本" / "Hillhouse Capital"
            / "GIC Private Limited (新加坡政府投资公司)" 等任意别名)
        asof: 业绩截止日 (基石锁定期内的回溯). 默认 today.
        min_confidence: < 此阈值视为未命中
        db_path: 测试注入

    Returns: 与 raw_names 等长的 CornerstoneInfo 列表; 未命中条目
        cornerstone_id=None / confidence=0.0 / 其他字段全 None.
        DB 不可用时整体返回未命中条目 (graceful degrade).
    """
    if not raw_names:
        return []
    if not is_available() and db_path is None:
        # 静默降级: 全部未命中
        return [
            CornerstoneInfo(
                raw_name=name, cornerstone_id=None, confidence=0.0,
                canonical_name=None, cornerstone_type=None,
                country_of_origin=None, aum_usd_latest=None,
                is_chinese=False, is_longterm=False,
                ipo_count_5y=None, avg_m6_return_5y=None,
                winrate_m6_5y=None, avg_d30_return_5y=None,
                lockup_discipline_score=None, sector_expertise=None,
            )
            for name in raw_names
        ]

    asof_str = asof.isoformat() if isinstance(asof, date) else (
        str(asof) if asof else date.today().isoformat()
    )
    out: list[CornerstoneInfo] = []
    try:
        with _connect(db_path) as conn:
            for name in raw_names:
                resolved = resolve_cornerstone_id(
                    conn, name, min_confidence=min_confidence,
                )
                if not resolved:
                    out.append(_unmatched(name))
                    continue
                cs_id, conf = resolved
                m_row = conn.execute(
                    "SELECT * FROM cornerstone_master WHERE cornerstone_id = ?",
                    (cs_id,),
                ).fetchone()
                p_row = conn.execute(
                    "SELECT * FROM cornerstone_performance_asof "
                    "WHERE cornerstone_id = ? AND as_of_date <= ? "
                    "ORDER BY as_of_date DESC LIMIT 1",
                    (cs_id, asof_str),
                ).fetchone()
                out.append(_to_info(name, cs_id, conf, m_row, p_row))
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant.cs] enrich_cornerstone_names 失败 (整体降级): {e}")
        return [_unmatched(name) for name in raw_names]
    return out


def render_enrichment_md(
    rows: list[CornerstoneInfo],
    *,
    show_unmatched: bool = True,
) -> str:
    """渲染成 markdown 表格 + 摘要, 喂给 CornerstoneAgent prompt 用."""
    if not rows:
        return "(无基石投资人数据)"

    matched = [r for r in rows if r.cornerstone_id is not None]
    n_total = len(rows)
    n_matched = len(matched)

    # 摘要
    lines = [
        f"### 基石投资人画像 (hkquant 解析)",
        "",
        f"- 招股书提取 **{n_total}** 家; hkquant 命中 **{n_matched}** 家",
    ]
    if matched:
        n_chinese = sum(1 for r in matched if r.is_chinese)
        n_longterm = sum(1 for r in matched if r.is_longterm)
        wrs = [r.winrate_m6_5y for r in matched if r.winrate_m6_5y is not None]
        m6s = [r.avg_m6_return_5y for r in matched if r.avg_m6_return_5y is not None]
        lds = [
            r.lockup_discipline_score for r in matched
            if r.lockup_discipline_score is not None
        ]
        if wrs:
            lines.append(f"- M6 胜率均值 (5y): **{sum(wrs)/len(wrs)*100:.1f}%**")
        if m6s:
            lines.append(
                f"- M6 平均回报 (5y): **{sum(m6s)/len(m6s)*100:+.1f}%**"
            )
        if lds:
            lines.append(
                f"- 锁定期纪律均值: **{sum(lds)/len(lds):.2f}** (0-1, 越高越优)"
            )
        lines.append(f"- 中资基石: {n_chinese}/{n_matched} 家")
        lines.append(f"- 长线锚定: {n_longterm}/{n_matched} 家 ⭐")

    # 详细表
    lines.append("")
    lines.append("| 招股书原文 | 解析名 | 类型 | 中资/长线 | 历史 IPO | M6 胜率 | M6 均值 | 锁定期纪律 | conf |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.cornerstone_id is None:
            if show_unmatched:
                lines.append(
                    f"| {r.raw_name} | (未命中) | — | — | — | — | — | — | — |"
                )
            continue
        cn = "✓" if r.is_chinese else "—"
        lt = "✓" if r.is_longterm else "—"
        lines.append(
            f"| {r.raw_name} | {r.canonical_name or '—'} | "
            f"{r.cornerstone_type or '—'} | {cn}/{lt} | "
            f"{r.ipo_count_5y if r.ipo_count_5y is not None else '—'} | "
            f"{f'{r.winrate_m6_5y * 100:.0f}%' if r.winrate_m6_5y is not None else '—'} | "
            f"{f'{r.avg_m6_return_5y * 100:+.0f}%' if r.avg_m6_return_5y is not None else '—'} | "
            f"{f'{r.lockup_discipline_score:.2f}' if r.lockup_discipline_score is not None else '—'} | "
            f"{r.confidence:.2f} |"
        )
    return "\n".join(lines)


# ---------- helpers ----------

def _unmatched(raw_name: str) -> CornerstoneInfo:
    return CornerstoneInfo(
        raw_name=raw_name, cornerstone_id=None, confidence=0.0,
        canonical_name=None, cornerstone_type=None,
        country_of_origin=None, aum_usd_latest=None,
        is_chinese=False, is_longterm=False,
        ipo_count_5y=None, avg_m6_return_5y=None,
        winrate_m6_5y=None, avg_d30_return_5y=None,
        lockup_discipline_score=None, sector_expertise=None,
    )


def _to_info(
    raw_name: str, cs_id: str, conf: float,
    m_row: sqlite3.Row | None, p_row: sqlite3.Row | None,
) -> CornerstoneInfo:
    m = dict(m_row) if m_row else {}
    p = dict(p_row) if p_row else {}
    return CornerstoneInfo(
        raw_name=raw_name, cornerstone_id=cs_id, confidence=round(conf, 3),
        canonical_name=m.get("canonical_name"),
        cornerstone_type=m.get("cornerstone_type"),
        country_of_origin=m.get("country_of_origin"),
        aum_usd_latest=m.get("aum_usd_latest"),
        is_chinese=bool(m.get("is_chinese", 0)),
        is_longterm=bool(m.get("is_longterm", 0)),
        ipo_count_5y=p.get("ipo_count_5y"),
        avg_m6_return_5y=p.get("avg_m6_return_5y"),
        winrate_m6_5y=p.get("winrate_m6_5y"),
        avg_d30_return_5y=p.get("avg_d30_return_5y"),
        lockup_discipline_score=p.get("lockup_discipline_score"),
        sector_expertise=p.get("sector_expertise"),
    )
