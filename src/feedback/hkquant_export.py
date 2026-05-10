"""T2.5: 把本项目 closed prediction + outcome 导出为 hkquant 兼容 SQLite.

设计原则:
- **永不写入用户的 nacs_real.db** (避免污染上游训练集)
- 默认写到独立文件 (cache_dir/hkquant_export_<ts>.db), schema 完全兼容
  hkquant.ipo_master + hkquant.ipo_returns
- 用户拿到导出文件后可手动 review, 再决定如何 merge:
    sqlite> ATTACH DATABASE 'export.db' AS exp;
    sqlite> INSERT OR IGNORE INTO ipo_master SELECT * FROM exp.ipo_master;
    sqlite> INSERT OR IGNORE INTO ipo_returns SELECT * FROM exp.ipo_returns;

字段映射 (Prediction + Outcome → hkquant):

  ipo_master:
    ipo_id              ← project_id
    stock_code          ← ticker
    company_name_zh     ← company_name (假定中文; 英文也直接放)
    listing_date        ← Outcome.final_listing_date
    listing_chapter     ← Prediction.listing_chapter (CHAPTER_MAP 翻译)
    gics_l2             ← THEME_TO_GICS_L2 反查 (industry_theme → 主推 GICS)
    offer_price_hkd     ← Outcome.ipo_actual_price_hkd
    offering_size_hkd   ← (空, 本项目不收集)
    cornerstone_total_hkd  ← Outcome.cornerstone_actual_amount_usd_million × FX
    lockup_months       ← 6 (港股惯例)
    data_quality_score  ← 0.7 (本项目导出, 标识非 hkquant 一手数据)
    data_source_notes   ← "tradingagent_hk export <decision_date>"

  ipo_returns:
    ipo_id              ← project_id
    return_d1_close     ← Outcome.d1_return
    return_d30          ← Outcome.d30_return
    return_m6           ← Outcome.d180_return    (6 月 ≈ 180 日)
    return_m12          ← Outcome.d365_return
    max_drawdown_m6     ← Outcome.max_drawdown_in_lockup_pct
    avg_daily_volume_hkd ← Outcome.avg_daily_turnover_hkd_m_d180 × 1e6
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.hkquant_client import THEME_TO_GICS_L2
from src.feedback.models import Outcome, Prediction
from src.feedback.store import FeedbackStore


# 本项目 listing_chapter (Main_Board_Standard / 18A / 18C / 19C / Secondary_Listing
# / Dual_Primary_AH / GEM / Unknown) → hkquant.ipo_master.listing_chapter (Main /
# 18A / 18C / GEM / ...). hkquant 实际取值待用户跑通后微调本表.
CHAPTER_MAP: dict[str, str] = {
    "Main_Board_Standard": "Main",
    "18A": "18A",
    "18C": "18C",
    "19C": "19C",
    "Secondary_Listing": "Secondary",
    "Dual_Primary_AH": "Dual_Primary",
    "GEM": "GEM",
    "Unknown": "Main",  # 兜底
}

USD_TO_HKD = 7.8  # 简化常量, 港币紧盯美元


# ============================================================================
# 字段映射纯函数 (易于单测)
# ============================================================================

def map_chapter(value: str | None) -> str:
    """tradingagent listing_chapter → hkquant 命名."""
    if not value:
        return "Main"
    return CHAPTER_MAP.get(value, "Main")


def map_industry_theme_to_gics_l2(industry_theme: str | None) -> str | None:
    """industry_theme → 主推 GICS L2 (取候选 set 第一个; 不命中返 None)."""
    if not industry_theme:
        return None
    candidates = THEME_TO_GICS_L2.get(industry_theme, ())
    return candidates[0] if candidates else None


def to_ipo_master_row(p: Prediction, o: Outcome | None) -> dict[str, Any] | None:
    """把 Prediction + Outcome 拼成 hkquant.ipo_master 行;
    没有 Outcome (未结算) 或 缺关键字段 (listing_date) 返 None.
    """
    if o is None or o.final_listing_date is None:
        return None

    listing_date = (
        o.final_listing_date.isoformat()
        if isinstance(o.final_listing_date, (date, datetime))
        else str(o.final_listing_date)
    )

    cs_total_hkd = None
    if o.cornerstone_actual_amount_usd_million is not None:
        cs_total_hkd = float(o.cornerstone_actual_amount_usd_million) * 1e6 * USD_TO_HKD

    decision_date_str = (
        p.decision_date.isoformat()
        if isinstance(p.decision_date, (date, datetime)) else str(p.decision_date)
    )

    return {
        "ipo_id": p.project_id,
        "stock_code": p.ticker,
        "company_name_zh": p.company_name,
        "company_name_en": "",
        "listing_date": listing_date,
        "pricing_date": None,
        "listing_chapter": map_chapter(p.listing_chapter),
        "is_a_h": 1 if p.has_a_share_listed else 0,
        "a_share_code": None,
        "gics_l2": map_industry_theme_to_gics_l2(p.industry_theme),
        "offer_price_hkd": o.ipo_actual_price_hkd,
        "offer_price_low": None,
        "offer_price_high": None,
        "offering_size_hkd": None,
        "pricing_in_range": None,
        "intl_oversub": None,
        "public_oversub": None,
        "clawback_triggered": None,
        "greenshoe_pct": None,
        "greenshoe_exercised": None,
        "sponsor_primary": None,
        "sponsor_tier": None,
        "joint_sponsor_count": 1,
        "auditor_tier": 1,
        "pe_at_offer": None,
        "pe_peer_median": None,
        "last_round_premium": None,
        "cornerstone_total_hkd": cs_total_hkd,
        "cornerstone_coverage": None,
        "cornerstone_count": None,
        "lockup_months": 6,
        "is_delisted": 0,
        "delisting_date": None,
        "is_acquired": 0,
        "data_quality_score": 0.7,  # 标识非一手数据
        "data_source_notes": f"tradingagent_hk export decision={decision_date_str}",
    }


def to_ipo_returns_row(p: Prediction, o: Outcome | None) -> dict[str, Any] | None:
    """Prediction + Outcome → hkquant.ipo_returns 行.

    全部 horizon 都 None 时返 None (没值导出无意义).
    """
    if o is None:
        return None
    fields = {
        "return_d1_close": o.d1_return,
        "return_d30": o.d30_return,
        "return_m6": o.d180_return,
        "return_m12": o.d365_return,
    }
    if all(v is None for v in fields.values()):
        return None

    avg_vol_hkd = None
    if o.avg_daily_turnover_hkd_m_d180 is not None:
        avg_vol_hkd = float(o.avg_daily_turnover_hkd_m_d180) * 1e6

    return {
        "ipo_id": p.project_id,
        "return_d1_close": fields["return_d1_close"],
        "return_d30": fields["return_d30"],
        "return_m3": None,                # 本项目无 d90 standardized horizon
        "return_m6": fields["return_m6"],
        "return_m12": fields["return_m12"],
        "return_unlock_d30": None,
        "return_unlock_d90": None,
        "max_drawdown_m6": o.max_drawdown_in_lockup_pct,
        "avg_daily_volume_hkd": avg_vol_hkd,
    }


# ============================================================================
# SQLite writer
# ============================================================================

_HKQUANT_EXPORT_SCHEMA = """
CREATE TABLE IF NOT EXISTS ipo_master (
    ipo_id TEXT PRIMARY KEY, stock_code TEXT NOT NULL,
    company_name_zh TEXT, company_name_en TEXT,
    listing_date DATE NOT NULL, pricing_date DATE,
    listing_chapter TEXT NOT NULL, is_a_h INTEGER DEFAULT 0,
    a_share_code TEXT, gics_l2 TEXT,
    offer_price_hkd REAL, offer_price_low REAL, offer_price_high REAL,
    offering_size_hkd REAL, pricing_in_range REAL,
    intl_oversub REAL, public_oversub REAL,
    clawback_triggered INTEGER, greenshoe_pct REAL,
    greenshoe_exercised INTEGER,
    sponsor_primary TEXT, sponsor_tier INTEGER,
    joint_sponsor_count INTEGER DEFAULT 1, auditor_tier INTEGER DEFAULT 1,
    pe_at_offer REAL, pe_peer_median REAL, last_round_premium REAL,
    cornerstone_total_hkd REAL, cornerstone_coverage REAL,
    cornerstone_count INTEGER, lockup_months INTEGER DEFAULT 6,
    is_delisted INTEGER DEFAULT 0, delisting_date DATE,
    is_acquired INTEGER DEFAULT 0,
    data_quality_score REAL DEFAULT 1.0, data_source_notes TEXT
);
CREATE TABLE IF NOT EXISTS ipo_returns (
    ipo_id TEXT PRIMARY KEY,
    return_d1_close REAL, return_d30 REAL, return_m3 REAL,
    return_m6 REAL, return_m12 REAL,
    return_unlock_d30 REAL, return_unlock_d90 REAL,
    max_drawdown_m6 REAL, avg_daily_volume_hkd REAL
);
"""


@dataclass
class ExportStats:
    predictions_scanned: int
    rows_master_written: int
    rows_returns_written: int
    skipped_no_outcome: int
    skipped_no_listing_date: int
    output_path: str


def export_to_hkquant_sqlite(
    out_path: str | Path,
    *,
    feedback_store: FeedbackStore | None = None,
    only_closed: bool = True,
) -> ExportStats:
    """导出本项目 prediction + outcome → 独立 SQLite (hkquant 兼容).

    Args:
        out_path: 输出 SQLite 绝对路径. 已存在会被追加 (INSERT OR REPLACE).
        feedback_store: 注入用; 默认新建一个连默认 feedback.sqlite
        only_closed: True (默认) 仅导出 status='closed' 的 prediction;
            False 导出所有有 outcome 的 (含 'open' 但已记录中期 outcome 的).

    Returns: ExportStats
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    own_store = feedback_store is None
    fs = feedback_store or FeedbackStore()
    try:
        # 拉所有 prediction
        if only_closed:
            preds = fs.list_predictions(status="closed", limit=10000)
        else:
            preds = fs.list_predictions(limit=10000)

        out_conn = sqlite3.connect(str(out_path))
        out_conn.executescript(_HKQUANT_EXPORT_SCHEMA)

        n_master = n_returns = 0
        skipped_no_outcome = skipped_no_date = 0
        for p in preds:
            # outcome 通过 prediction id 拉
            pid = fs._conn.execute(
                "SELECT id FROM predictions WHERE project_id = ?",
                (p.project_id,),
            ).fetchone()
            if pid is None:
                continue
            o = fs.latest_outcome(int(pid["id"]))
            if o is None:
                skipped_no_outcome += 1
                continue
            m_row = to_ipo_master_row(p, o)
            if m_row is None:
                skipped_no_date += 1
                continue
            cols = list(m_row.keys())
            placeholders = ",".join(["?"] * len(cols))
            out_conn.execute(
                f"INSERT OR REPLACE INTO ipo_master ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                [m_row[c] for c in cols],
            )
            n_master += 1

            r_row = to_ipo_returns_row(p, o)
            if r_row is not None:
                cols = list(r_row.keys())
                placeholders = ",".join(["?"] * len(cols))
                out_conn.execute(
                    f"INSERT OR REPLACE INTO ipo_returns ({','.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [r_row[c] for c in cols],
                )
                n_returns += 1

        out_conn.commit()
        out_conn.close()

        stats = ExportStats(
            predictions_scanned=len(preds),
            rows_master_written=n_master,
            rows_returns_written=n_returns,
            skipped_no_outcome=skipped_no_outcome,
            skipped_no_listing_date=skipped_no_date,
            output_path=str(out_path),
        )
        logger.info(
            f"[hkquant_export] {stats.predictions_scanned} 个 prediction → "
            f"ipo_master {stats.rows_master_written} / "
            f"ipo_returns {stats.rows_returns_written} 行 → {stats.output_path}"
        )
        return stats
    finally:
        if own_store:
            fs.close()


# ============================================================================
# CLI
# ============================================================================

def _default_export_path() -> Path:
    from config import get_settings
    s = get_settings()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return s.cache_dir / f"hkquant_export_{ts}.db"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hkquant_export",
        description=(
            "把 tradingagent_hk 的 closed prediction + outcome 导出为 hkquant "
            "兼容 SQLite. 默认仅导出 status='closed' 的项目."
        ),
    )
    ap.add_argument("--out", default=None,
                    help="输出 SQLite 路径; 默认 cache_dir/hkquant_export_<ts>.db")
    ap.add_argument("--all", action="store_true",
                    help="导出所有有 outcome 的 prediction (含 open 状态)")
    args = ap.parse_args(argv)

    out_path = Path(args.out) if args.out else _default_export_path()
    stats = export_to_hkquant_sqlite(out_path, only_closed=not args.all)

    print(
        f"[OK] 扫 {stats.predictions_scanned} prediction → "
        f"ipo_master {stats.rows_master_written} 行, "
        f"ipo_returns {stats.rows_returns_written} 行 → {stats.output_path}"
    )
    if stats.skipped_no_outcome:
        print(f"  跳过 {stats.skipped_no_outcome} 个 (无 outcome)")
    if stats.skipped_no_listing_date:
        print(f"  跳过 {stats.skipped_no_listing_date} 个 (outcome 缺 listing_date)")
    print(
        "\n下一步 (在 hkquant 项目里手动 merge):\n"
        f"  sqlite3 nacs_real.db\n"
        f"  > ATTACH DATABASE '{stats.output_path}' AS exp;\n"
        f"  > INSERT OR IGNORE INTO ipo_master SELECT * FROM exp.ipo_master;\n"
        f"  > INSERT OR IGNORE INTO ipo_returns SELECT * FROM exp.ipo_returns;"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
