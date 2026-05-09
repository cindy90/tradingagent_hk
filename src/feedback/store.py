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
    decision_weights_json TEXT,
    weighted_total_score REAL,
    weighted_to_recommendation_mapping TEXT,

    listing_chapter TEXT DEFAULT 'Unknown',
    size_tier TEXT DEFAULT 'Unknown',
    industry_theme TEXT DEFAULT 'Other',
    has_wvr INTEGER DEFAULT 0,
    has_a_share_listed INTEGER DEFAULT 0,

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

    weight_calibrations_json TEXT,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scores_prediction ON scores(prediction_id);
"""

_JSON_FIELDS_PRED = {
    "key_supports", "key_risks", "deal_conditions", "monitoring_kpis",
    "agent_score_cards", "reviewer_scores", "diversity_variants",
    "model_tier_models", "cogalpha_features_used",
    "decision_weights",
}
_JSON_FIELDS_OUT = {"notable_events"}
_JSON_FIELDS_SCR = {"per_agent_quality", "error_root_causes", "weight_calibrations"}


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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """轻量迁移：对老 DB 增加新列（不会冲突, 已存在时忽略）。"""
        existing_score_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(scores)").fetchall()
        }
        if "weight_calibrations_json" not in existing_score_cols:
            try:
                self._conn.execute("ALTER TABLE scores ADD COLUMN weight_calibrations_json TEXT")
                logger.info("[migrate] scores 表添加列 weight_calibrations_json")
            except sqlite3.OperationalError as e:
                logger.debug(f"[migrate] 添加 weight_calibrations_json 失败: {e}")

        existing_pred_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for col, ddl in [
            ("decision_weights_json", "TEXT"),
            ("weighted_total_score", "REAL"),
            ("weighted_to_recommendation_mapping", "TEXT"),
            ("listing_chapter", "TEXT DEFAULT 'Unknown'"),
            ("size_tier", "TEXT DEFAULT 'Unknown'"),
            ("industry_theme", "TEXT DEFAULT 'Other'"),
            ("has_wvr", "INTEGER DEFAULT 0"),
            ("has_a_share_listed", "INTEGER DEFAULT 0"),
        ]:
            if col not in existing_pred_cols:
                try:
                    self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {ddl}")
                    logger.info(f"[migrate] predictions 表添加列 {col}")
                except sqlite3.OperationalError as e:
                    logger.debug(f"[migrate] 添加 {col} 失败: {e}")

    def close(self) -> None:
        self._conn.close()

    # ---------- predictions ----------

    def save_prediction(self, p: Prediction, *, upsert: bool = True) -> int:
        """落库 prediction. 默认 upsert: 同 project_id 已存在则 UPDATE。

        Args:
            upsert: True (默认) = 已存在则更新; False = 已存在则跳过。
        """
        d = p.model_dump()
        cols, vals, placeholders = [], [], []
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
            if "UNIQUE constraint failed" not in str(e):
                raise
            existing = self._conn.execute(
                "SELECT id FROM predictions WHERE project_id=?", (p.project_id,)
            ).fetchone()
            existing_id = existing["id"] if existing else -1
            if not upsert:
                logger.warning(f"project_id={p.project_id} 已存在，跳过插入")
                return existing_id
            # UPDATE: 不动 id, 其他字段全部覆盖
            set_clause = ",".join(f"{c}=?" for c in cols)
            self._conn.execute(
                f"UPDATE predictions SET {set_clause} WHERE id=?",
                vals + [existing_id],
            )
            self._conn.commit()
            logger.info(f"prediction 更新: id={existing_id} project={p.project_id}")
            return existing_id

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

    # ---------- weight calibration priors (Phase B) ----------

    def get_weight_calibration_priors(
        self,
        industry: str,
        recommendation: str | None = None,
        *,
        listing_chapter: str | None = None,
        size_tier: str | None = None,
        min_samples: int = 5,
        max_samples: int = 30,
    ) -> dict[str, Any]:
        """聚合历史 closed prediction 的权重校准建议, 给新项目当 prior。

        三维度匹配（v2: 加 listing_chapter + size_tier）:
          1. industry 模糊匹配 (LIKE 核心关键词)
          2. listing_chapter 严格匹配 (例 18A 项目不混用主板项目的 calibration)
          3. size_tier 严格匹配 (Small/Mid/Large/Mega 不混)
          4. recommendation 可选过滤

        若指定 listing_chapter / size_tier 但样本不足, 自动逐级 fallback:
          (industry + chapter + size) → (industry + chapter) → (industry) → 空

        Returns: {"sample_size": N, "match_level": "...", "calibrations": [...]}
        """
        # 模糊行业匹配 - 取核心 2-3 字
        industry_keywords = self._derive_industry_keywords(industry)
        if not industry_keywords:
            return {"sample_size": 0, "calibrations": [],
                    "industry_filter": industry, "recommendation_filter": recommendation}

        # 构 SQL: 必须 closed (有 outcome 才有意义) + 有 calibration
        like_clauses = " OR ".join(["p.industry LIKE ?"] * len(industry_keywords))
        like_args = [f"%{k}%" for k in industry_keywords]

        # 三维度 fallback (最严格 → 最宽), 第一个命中样本数 ≥ min_samples 的层即采用
        fallback_levels: list[tuple[str, list[str], list[Any]]] = []
        # Level 1: industry + listing_chapter + size_tier
        if listing_chapter and size_tier and listing_chapter != "Unknown" and size_tier != "Unknown":
            fallback_levels.append((
                "industry+chapter+size_tier",
                [f"({like_clauses})", "p.listing_chapter = ?", "p.size_tier = ?"],
                like_args + [listing_chapter, size_tier],
            ))
        # Level 2: industry + listing_chapter
        if listing_chapter and listing_chapter != "Unknown":
            fallback_levels.append((
                "industry+chapter",
                [f"({like_clauses})", "p.listing_chapter = ?"],
                like_args + [listing_chapter],
            ))
        # Level 3: industry + size_tier
        if size_tier and size_tier != "Unknown":
            fallback_levels.append((
                "industry+size_tier",
                [f"({like_clauses})", "p.size_tier = ?"],
                like_args + [size_tier],
            ))
        # Level 4: industry only
        fallback_levels.append((
            "industry",
            [f"({like_clauses})"],
            list(like_args),
        ))

        rows: list = []
        match_level = "industry"
        for level_name, where_parts, level_args in fallback_levels:
            full_args = list(level_args)
            where_clause = " AND ".join(where_parts)
            sql = f"""
                SELECT p.industry, p.recommendation, p.listing_chapter, p.size_tier,
                       s.weight_calibrations_json
                FROM scores s
                JOIN predictions p ON p.id = s.prediction_id
                WHERE {where_clause}
                  AND s.weight_calibrations_json IS NOT NULL
                  AND s.weight_calibrations_json != '[]'
            """
            if recommendation:
                sql += " AND p.recommendation = ?"
                full_args.append(recommendation)
            sql += " ORDER BY s.score_date DESC LIMIT ?"
            full_args.append(max_samples)
            try:
                rows = self._conn.execute(sql, full_args).fetchall()
            except sqlite3.OperationalError:
                # 老 DB 无 listing_chapter / size_tier 列, 跳过该层级
                continue
            if rows and len(rows) >= min_samples:
                match_level = level_name
                break
            elif rows and len(rows) > 0 and level_name == fallback_levels[-1][0]:
                # 最后一层有数据但 < min_samples, 仍记录
                match_level = level_name
                break

        if not rows:
            return {"sample_size": 0, "calibrations": [],
                    "industry_filter": industry, "recommendation_filter": recommendation,
                    "listing_chapter_filter": listing_chapter, "size_tier_filter": size_tier,
                    "match_level": "no_match"}

        # 按 factor 聚合
        from collections import defaultdict
        deltas_by_factor: dict[str, list[float]] = defaultdict(list)
        rationales_by_factor: dict[str, list[str]] = defaultdict(list)

        for row in rows:
            try:
                cals = json.loads(row["weight_calibrations_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(cals, list):
                continue
            for c in cals:
                if not isinstance(c, dict):
                    continue
                factor = c.get("factor")
                if not factor:
                    continue
                delta = c.get("delta")
                if delta is None:
                    actual = c.get("actual_weight_used", 0)
                    suggested = c.get("suggested_weight", 0)
                    delta = round(suggested - actual, 4) if (actual or suggested) else 0
                try:
                    deltas_by_factor[factor].append(float(delta))
                except (TypeError, ValueError):
                    continue
                rationale = (c.get("rationale") or "").strip()
                if rationale:
                    rationales_by_factor[factor].append(rationale)

        n_samples = len(rows)
        if n_samples < min_samples:
            # 样本量不足, 返回但 calibrations 为空, prompt 注入逻辑会跳过
            return {
                "sample_size": n_samples, "calibrations": [],
                "industry_filter": industry, "recommendation_filter": recommendation,
                "listing_chapter_filter": listing_chapter, "size_tier_filter": size_tier,
                "match_level": match_level,
                "min_samples_required": min_samples,
                "note": f"历史样本 {n_samples} < 最低门槛 {min_samples} "
                f"(命中层级: {match_level}), 不注入校准",
            }

        # 聚合每个 factor 的平均偏差 + min/max + rationale 示例
        calibrations = []
        for factor, deltas in deltas_by_factor.items():
            if not deltas:
                continue
            avg_delta = round(sum(deltas) / len(deltas), 4)
            calibrations.append({
                "factor": factor,
                "avg_delta": avg_delta,
                "samples": len(deltas),
                "min_delta": round(min(deltas), 4),
                "max_delta": round(max(deltas), 4),
                # 取最近 2 条 rationale 作为示例
                "rationale_examples": rationales_by_factor.get(factor, [])[:2],
            })
        # 按 |avg_delta| 排序, 偏差大的因子放前面 (LLM 更关注)
        calibrations.sort(key=lambda x: abs(x["avg_delta"]), reverse=True)

        return {
            "sample_size": n_samples,
            "industry_filter": industry,
            "recommendation_filter": recommendation,
            "listing_chapter_filter": listing_chapter,
            "size_tier_filter": size_tier,
            "match_level": match_level,
            "calibrations": calibrations,
        }

    @staticmethod
    def _derive_industry_keywords(industry: str) -> list[str]:
        """从 industry 字段抽核心关键词, 用于模糊匹配。"""
        if not industry:
            return []
        parts = [p.strip() for p in industry.replace("、", "/").replace(",", "/").split("/")]
        out: set[str] = set()
        for p in parts:
            if not p:
                continue
            # 取 ≥ 2 字的子串作为关键词
            if len(p) >= 2:
                out.add(p)
            for sub in p.split():
                if len(sub) >= 2:
                    out.add(sub)
        return list(out)

    # ---------- stats ----------

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
