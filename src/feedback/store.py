"""SQLite 持久化层。

设计:
- 三张表: predictions / outcomes / scores（一对多关系）
- JSON 字段用 TEXT 存（SQLite 1.4.0+ 原生支持 JSON 函数）
- 每个 ticker 可以有多次 prediction（不同时间多次跑分析）
- 每个 prediction 可以有多条 outcome（结果分阶段更新）
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import get_settings

from .models import Outcome, Prediction, Score

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    industry TEXT,
    decision_date TEXT NOT NULL,

    recommendation TEXT,
    confidence TEXT,
    valuation_low REAL,
    valuation_mid REAL,
    valuation_high REAL,
    anchor_method TEXT,
    anchor_logic TEXT,
    ipo_pricing_view TEXT,
    suggested_amount_low_usd_m REAL,
    suggested_amount_high_usd_m REAL,
    key_supports_json TEXT,
    key_risks_json TEXT,
    deal_conditions_json TEXT,
    monitoring_kpis_json TEXT,

    agent_score_cards_json TEXT,
    reviewer_scores_json TEXT,
    diversity_variants_json TEXT,

    model_provider TEXT,
    model_tier_models_json TEXT,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cache_read_tokens INTEGER DEFAULT 0,
    estimated_cost_cny REAL,
    cogalpha_features_used_json TEXT,

    reports_dir_path TEXT,
    status TEXT DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    recorded_date TEXT NOT NULL,

    ipo_actual_price_hkd REAL,
    ipo_actual_marketcap_hkd_billion REAL,
    final_listing_date TEXT,

    d1_return REAL,
    d30_return REAL,
    d90_return REAL,
    d180_return REAL,
    d365_return REAL,
    d180_alpha_vs_hsi REAL,
    d365_alpha_vs_hsi REAL,

    was_broken_ipo_d1 INTEGER,
    was_broken_ipo_d180 INTEGER,
    max_drawdown_in_lockup_pct REAL,
    min_price_in_lockup_hkd REAL,

    avg_daily_turnover_hkd_m_d180 REAL,
    free_float_pct REAL,

    cornerstone_actual_amount_usd_million REAL,
    cornerstone_lockup_end_date TEXT,
    cornerstone_realized_return_pct REAL,

    user_notes TEXT,
    notable_events_json TEXT,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON outcomes(prediction_id);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    score_date TEXT NOT NULL,

    recommendation_score REAL,
    valuation_within_range INTEGER,
    valuation_error_pct REAL,

    risks_total_count INTEGER,
    risks_realized_count INTEGER,
    unforeseen_risks_count INTEGER,

    per_agent_quality_json TEXT,
    confidence_calibration_delta REAL,

    postmortem_memo TEXT,
    error_root_causes_json TEXT,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scores_prediction ON scores(prediction_id);
"""

_JSON_FIELDS_PRED = {
    "key_supports", "key_risks", "deal_conditions", "monitoring_kpis",
    "agent_score_cards", "reviewer_scores", "diversity_variants",
    "model_tier_models", "cogalpha_features_used",
}
_JSON_FIELDS_OUT = {"notable_events"}
_JSON_FIELDS_SCR = {"per_agent_quality", "error_root_causes"}


def _to_db_value(v: Any) -> Any:
    if isinstance(v, (datetime,)):
        return v.isoformat()
    if hasattr(v, "isoformat"):  # date
        return v.isoformat()
    if isinstance(v, bool):
        return int(v)
    return v


class FeedbackStore:
    def __init__(self, db_path: str | Path | None = None):
        s = get_settings()
        self.db_path = Path(db_path) if db_path else (s.cache_dir / "feedback.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------- predictions ----------

    def save_prediction(self, p: Prediction) -> int:
        d = p.model_dump()
        cols = []
        vals = []
        placeholders = []
        for k, v in d.items():
            if k in _JSON_FIELDS_PRED:
                cols.append(f"{k}_json")
                vals.append(json.dumps(v, ensure_ascii=False, default=str))
            else:
                cols.append(k)
                vals.append(_to_db_value(v))
            placeholders.append("?")
        sql = f"INSERT INTO predictions ({','.join(cols)}) VALUES ({','.join(placeholders)})"
        try:
            cur = self._conn.execute(sql, vals)
            self._conn.commit()
            logger.info(f"prediction 落库: id={cur.lastrowid} project={p.project_id}")
            return cur.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"project_id={p.project_id} 已存在，跳过插入")
                row = self._conn.execute(
                    "SELECT id FROM predictions WHERE project_id=?", (p.project_id,)
                ).fetchone()
                return row["id"] if row else -1
            raise

    def get_prediction(self, pred_id: int) -> Prediction | None:
        row = self._conn.execute(
            "SELECT * FROM predictions WHERE id=?", (pred_id,)
        ).fetchone()
        return _row_to_prediction(row) if row else None

    def get_prediction_by_project(self, project_id: str) -> Prediction | None:
        row = self._conn.execute(
            "SELECT * FROM predictions WHERE project_id=?", (project_id,)
        ).fetchone()
        return _row_to_prediction(row) if row else None

    def list_predictions(
        self,
        status: str | None = None,
        ticker: str | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        where = []
        args: list[Any] = []
        if status:
            where.append("status = ?")
            args.append(status)
        if ticker:
            where.append("ticker = ?")
            args.append(ticker)
        sql = "SELECT * FROM predictions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY decision_date DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(sql, args).fetchall()
        return [p for p in (_row_to_prediction(r) for r in rows) if p]

    def update_status(self, pred_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE predictions SET status=? WHERE id=?", (status, pred_id)
        )
        self._conn.commit()

    # ---------- outcomes ----------

    def record_outcome(self, o: Outcome) -> int:
        d = o.model_dump()
        cols = []
        vals = []
        placeholders = []
        for k, v in d.items():
            if k in _JSON_FIELDS_OUT:
                cols.append(f"{k}_json")
                vals.append(json.dumps(v, ensure_ascii=False, default=str))
            else:
                cols.append(k)
                vals.append(_to_db_value(v))
            placeholders.append("?")
        sql = f"INSERT INTO outcomes ({','.join(cols)}) VALUES ({','.join(placeholders)})"
        cur = self._conn.execute(sql, vals)
        self._conn.commit()
        logger.info(f"outcome 落库: id={cur.lastrowid} prediction_id={o.prediction_id}")
        return cur.lastrowid

    def latest_outcome(self, pred_id: int) -> Outcome | None:
        row = self._conn.execute(
            "SELECT * FROM outcomes WHERE prediction_id=? ORDER BY recorded_date DESC LIMIT 1",
            (pred_id,),
        ).fetchone()
        return _row_to_outcome(row) if row else None

    # ---------- scores ----------

    def save_score(self, s: Score) -> int:
        d = s.model_dump()
        cols = []
        vals = []
        placeholders = []
        for k, v in d.items():
            if k in _JSON_FIELDS_SCR:
                cols.append(f"{k}_json")
                vals.append(json.dumps(v, ensure_ascii=False, default=str))
            else:
                cols.append(k)
                vals.append(_to_db_value(v))
            placeholders.append("?")
        sql = f"INSERT INTO scores ({','.join(cols)}) VALUES ({','.join(placeholders)})"
        cur = self._conn.execute(sql, vals)
        self._conn.commit()
        return cur.lastrowid

    def latest_score(self, pred_id: int) -> Score | None:
        row = self._conn.execute(
            "SELECT * FROM scores WHERE prediction_id=? ORDER BY score_date DESC LIMIT 1",
            (pred_id,),
        ).fetchone()
        return _row_to_score(row) if row else None

    # ---------- 统计 ----------

    def stats_summary(self) -> dict[str, Any]:
        n_pred = self._conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        n_open = self._conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE status='open'"
        ).fetchone()["c"]
        n_closed = self._conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE status='closed'"
        ).fetchone()["c"]
        n_outcomes = self._conn.execute("SELECT COUNT(*) AS c FROM outcomes").fetchone()["c"]
        n_scores = self._conn.execute("SELECT COUNT(*) AS c FROM scores").fetchone()["c"]
        return {
            "predictions_total": n_pred,
            "predictions_open": n_open,
            "predictions_closed": n_closed,
            "outcomes_total": n_outcomes,
            "scores_total": n_scores,
        }

    def export_predictions_with_outcomes(self) -> list[dict]:
        """LEFT JOIN 导出，给 csv export / 外部分析用。"""
        rows = self._conn.execute(
            """
            SELECT p.*, o.d1_return, o.d180_return, o.d365_return,
                   o.was_broken_ipo_d1, o.cornerstone_realized_return_pct,
                   s.recommendation_score, s.valuation_error_pct
            FROM predictions p
            LEFT JOIN (
                SELECT * FROM outcomes
                WHERE id IN (SELECT MAX(id) FROM outcomes GROUP BY prediction_id)
            ) o ON p.id = o.prediction_id
            LEFT JOIN (
                SELECT * FROM scores
                WHERE id IN (SELECT MAX(id) FROM scores GROUP BY prediction_id)
            ) s ON p.id = s.prediction_id
            ORDER BY p.decision_date DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- row → model 反序列化 ----------

def _row_to_prediction(row: sqlite3.Row) -> Prediction | None:
    if row is None:
        return None
    d = dict(row)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.endswith("_json") and v is not None:
            out[k[:-5]] = json.loads(v)
        elif k.endswith("_json"):
            out[k[:-5]] = {} if k[:-5] in {"agent_score_cards", "reviewer_scores",
                                            "diversity_variants", "model_tier_models"} else []
        else:
            out[k] = v
    out.pop("id", None)
    return Prediction.model_validate(out)


def _row_to_outcome(row: sqlite3.Row) -> Outcome | None:
    if row is None:
        return None
    d = dict(row)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.endswith("_json") and v is not None:
            out[k[:-5]] = json.loads(v)
        elif k.endswith("_json"):
            out[k[:-5]] = []
        elif k in ("was_broken_ipo_d1", "was_broken_ipo_d180") and v is not None:
            out[k] = bool(v)
        else:
            out[k] = v
    out.pop("id", None)
    return Outcome.model_validate(out)


def _row_to_score(row: sqlite3.Row) -> Score | None:
    if row is None:
        return None
    d = dict(row)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.endswith("_json") and v is not None:
            out[k[:-5]] = json.loads(v)
        elif k.endswith("_json"):
            out[k[:-5]] = [] if k[:-5] == "error_root_causes" else {}
        elif k == "valuation_within_range" and v is not None:
            out[k] = bool(v)
        else:
            out[k] = v
    out.pop("id", None)
    return Score.model_validate(out)
