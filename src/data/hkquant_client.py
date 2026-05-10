"""hkquant (NACS v8) SQLite 只读 adapter.

接 cindy90/hkquant 项目的 nacs_real.db (本机绝对路径, env: HKQUANT_DB_PATH).
本模块是**只读** adapter — 不会写入 hkquant DB. schema 各自维护两份, 字段
变更时本文件需手动跟进.

提供给本项目的能力 (T1):
- 同 industry_theme/GICS L2 已上市港股 IPO peer + 首日/D30/M6 回报 (历史回溯)
- 过去 12 月同主题 IPO 数 (边际稀缺度)
- 同主题最近 IPO 首日开盘均值 (打新情绪锚)
- 月度市场环境 (HSI vol/估值, HK IPO 30d 破发率, 南下资金) — DecisionAgent regime gate

Schema 来源: https://github.com/cindy90/hkquant/blob/main/src/data/schema.py
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from config import get_settings


# ============================================================================
# industry_theme (本项目 12 枚举) → hkquant gics_l2 候选值 映射
# ============================================================================
# hkquant.ipo_master.gics_l2 是 GICS Level 2 字符串 (具体取值取决于 hkquant 入库时
# 用的命名). 此处给出按 GICS 标准 L2 命名的候选 set; 用户首次跑通后可据实际
# DB 取值微调本表.
THEME_TO_GICS_L2: dict[str, tuple[str, ...]] = {
    "Tech_AI_Semi": (
        "Software & Services",
        "Semiconductors & Semiconductor Equipment",
        "Technology Hardware & Equipment",
    ),
    "Bio_Pharma": (
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Biotechnology",
    ),
    "Med_Device": (
        "Health Care Equipment & Services",
        "Health Care Equipment & Supplies",
    ),
    "Robotics_Automation": (
        "Capital Goods",
        "Machinery",
    ),
    "New_Energy": (
        "Utilities",
        "Energy",
        "Automobiles & Components",  # 新能源车产业链
    ),
    "Advanced_Materials": (
        "Materials",
        "Chemicals",
    ),
    "Consumer": (
        "Consumer Discretionary Distribution & Retail",
        "Consumer Services",
        "Consumer Staples Distribution & Retail",
        "Consumer Durables & Apparel",
        "Food, Beverage & Tobacco",
        "Household & Personal Products",
    ),
    "Financial": (
        "Banks",
        "Financial Services",
        "Insurance",
    ),
    "Real_Estate": (
        "Real Estate",
        "Equity Real Estate Investment Trusts (REITs)",
    ),
    "Industrial": (
        "Capital Goods",
        "Commercial & Professional Services",
        "Transportation",
    ),
    "Healthcare": (
        "Health Care Equipment & Services",
    ),
    "Other": (),
}


# ============================================================================
# 数据类型
# ============================================================================

@dataclass
class HKQuantPeer:
    """hkquant.ipo_master + ipo_returns join 后的单家 peer 视图."""
    stock_code: str
    name: str
    listing_date: str            # ISO YYYY-MM-DD
    listing_chapter: str         # Main / 18A / 18C / GEM 等
    gics_l2: str | None
    offer_price_hkd: float | None
    offering_size_hkd: float | None
    return_d1_close: float | None    # 首日 close 回报 (vs offer_price)
    return_d30: float | None
    return_m6: float | None
    return_m12: float | None
    avg_daily_volume_hkd: float | None  # 单位 HKD
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class HKMarketEnv:
    """hkquant.market_environment_cache 单月快照."""
    asof_month: str                          # ISO YYYY-MM-01
    hsi_60d_return: float | None
    hsi_60d_vol_annualized: float | None
    hsi_60d_vol_pct_rank: float | None
    hsi_valuation_pct: float | None
    hk_ipo_30d_avg_d30: float | None
    hk_ipo_30d_breakage_rate: float | None
    southbound_30d_net_normalized: float | None
    sector_60d_vol_annualized: float | None


# ============================================================================
# 连接管理
# ============================================================================

def get_hkquant_db_path() -> Path | None:
    """返回 hkquant DB 绝对路径; 未配置或文件缺失则 None."""
    raw = (get_settings().hkquant_db_path or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.exists():
        logger.warning(f"[hkquant] HKQUANT_DB_PATH={p} 不存在, hkquant 接入降级")
        return None
    return p


def is_available() -> bool:
    return get_hkquant_db_path() is not None


@contextmanager
def _connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """以**只读** mode 打开 hkquant DB. URI mode 防误写."""
    p = Path(db_path) if db_path else get_hkquant_db_path()
    if p is None:
        raise FileNotFoundError("hkquant DB 路径未配置 (HKQUANT_DB_PATH)")
    uri = f"file:{p.absolute()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# 查询: GICS L2 维度的同主题 peer / 边际稀缺度
# ============================================================================

def _gics_filter_for_theme(industry_theme: str | None) -> tuple[str, ...]:
    if not industry_theme:
        return ()
    return THEME_TO_GICS_L2.get(industry_theme, ())


def _row_to_peer(row: sqlite3.Row) -> HKQuantPeer:
    d = dict(row)
    return HKQuantPeer(
        stock_code=d.get("stock_code", ""),
        name=d.get("company_name_zh") or d.get("company_name_en") or "",
        listing_date=str(d.get("listing_date", "")),
        listing_chapter=d.get("listing_chapter", ""),
        gics_l2=d.get("gics_l2"),
        offer_price_hkd=d.get("offer_price_hkd"),
        offering_size_hkd=d.get("offering_size_hkd"),
        return_d1_close=d.get("return_d1_close"),
        return_d30=d.get("return_d30"),
        return_m6=d.get("return_m6"),
        return_m12=d.get("return_m12"),
        avg_daily_volume_hkd=d.get("avg_daily_volume_hkd"),
        raw=d,
    )


def get_listed_peers_by_theme(
    industry_theme: str,
    *,
    asof_date: date | str | None = None,
    listing_chapter: str | None = None,
    lookback_years: int = 5,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[HKQuantPeer]:
    """按本项目 industry_theme → GICS L2 候选集, 拉同主题已上市 IPO + 历史回报.

    Args:
        industry_theme: 本项目 12 枚举之一
        asof_date: 截止日 (排除此日之后的 IPO); 默认今日
        listing_chapter: 仅限定章节 (Main / 18A / 18C); None = 不限
        lookback_years: 仅看过去 N 年内的 IPO (避免 5+ 年前老股噪音)
        limit: 返回家数上限
        db_path: 测试注入用; 默认读 settings

    Returns: 按 listing_date 倒序的 HKQuantPeer 列表; 未配置 / DB 不可用时 []
    """
    if not is_available() and db_path is None:
        return []
    gics = _gics_filter_for_theme(industry_theme)
    if not gics:
        return []
    asof = _to_date_str(asof_date) or date.today().isoformat()
    cutoff = (datetime.fromisoformat(asof).date()
              - timedelta(days=lookback_years * 365)).isoformat()

    placeholders = ",".join(["?"] * len(gics))
    sql = f"""
        SELECT
            m.stock_code, m.company_name_zh, m.company_name_en,
            m.listing_date, m.listing_chapter, m.gics_l2,
            m.offer_price_hkd, m.offering_size_hkd,
            r.return_d1_close, r.return_d30, r.return_m6, r.return_m12,
            r.avg_daily_volume_hkd
        FROM ipo_master m
        LEFT JOIN ipo_returns r ON r.ipo_id = m.ipo_id
        WHERE m.gics_l2 IN ({placeholders})
          AND m.listing_date <= ?
          AND m.listing_date >= ?
          AND COALESCE(m.is_delisted, 0) = 0
    """
    params: list[Any] = list(gics) + [asof, cutoff]
    if listing_chapter:
        sql += " AND m.listing_chapter = ?"
        params.append(listing_chapter)
    sql += " ORDER BY m.listing_date DESC LIMIT ?"
    params.append(limit)

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_listed_peers_by_theme 失败: {e}")
        return []
    return [_row_to_peer(r) for r in rows]


def get_recent_ipo_count_in_theme(
    industry_theme: str,
    *,
    asof_date: date | str | None = None,
    lookback_days: int = 365,
    db_path: Path | str | None = None,
) -> int:
    """过去 N 天同 industry_theme IPO 数 (边际稀缺度递减信号)."""
    if not is_available() and db_path is None:
        return 0
    gics = _gics_filter_for_theme(industry_theme)
    if not gics:
        return 0
    asof = _to_date_str(asof_date) or date.today().isoformat()
    start = (datetime.fromisoformat(asof).date()
             - timedelta(days=lookback_days)).isoformat()
    placeholders = ",".join(["?"] * len(gics))
    sql = (
        f"SELECT COUNT(*) AS n FROM ipo_master "
        f"WHERE gics_l2 IN ({placeholders}) "
        f"AND listing_date BETWEEN ? AND ?"
    )
    try:
        with _connect(db_path) as conn:
            row = conn.execute(sql, list(gics) + [start, asof]).fetchone()
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_recent_ipo_count_in_theme 失败: {e}")
        return 0
    return int(row["n"] or 0)


def get_avg_first_day_return_in_theme(
    industry_theme: str,
    *,
    asof_date: date | str | None = None,
    lookback_days: int = 365,
    min_samples: int = 3,
    db_path: Path | str | None = None,
) -> float | None:
    """过去 N 天同主题 IPO 首日 close 回报均值 (%). 样本 < min_samples 返 None."""
    if not is_available() and db_path is None:
        return None
    gics = _gics_filter_for_theme(industry_theme)
    if not gics:
        return None
    asof = _to_date_str(asof_date) or date.today().isoformat()
    start = (datetime.fromisoformat(asof).date()
             - timedelta(days=lookback_days)).isoformat()
    placeholders = ",".join(["?"] * len(gics))
    sql = f"""
        SELECT r.return_d1_close
        FROM ipo_master m
        JOIN ipo_returns r ON r.ipo_id = m.ipo_id
        WHERE m.gics_l2 IN ({placeholders})
          AND m.listing_date BETWEEN ? AND ?
          AND r.return_d1_close IS NOT NULL
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(sql, list(gics) + [start, asof]).fetchall()
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_avg_first_day_return_in_theme 失败: {e}")
        return None
    vals = [r["return_d1_close"] for r in rows if r["return_d1_close"] is not None]
    if len(vals) < min_samples:
        return None
    # ipo_returns.return_d1_close 是小数 (0.05 = +5%); 输出 % 与本项目对齐
    return round(sum(vals) / len(vals) * 100.0, 2)


# ============================================================================
# 查询: 月度市场环境 (regime gate)
# ============================================================================

def get_market_environment_at(
    asof_date: date | str | None = None,
    *,
    db_path: Path | str | None = None,
) -> HKMarketEnv | None:
    """读 market_environment_cache, 取 ≤ asof 的最新月份快照."""
    if not is_available() and db_path is None:
        return None
    asof = _to_date_str(asof_date) or date.today().isoformat()
    sql = (
        "SELECT * FROM market_environment_cache "
        "WHERE asof_month <= ? ORDER BY asof_month DESC LIMIT 1"
    )
    try:
        with _connect(db_path) as conn:
            row = conn.execute(sql, (asof,)).fetchone()
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_market_environment_at 失败: {e}")
        return None
    if not row:
        return None
    d = dict(row)
    return HKMarketEnv(
        asof_month=str(d.get("asof_month", "")),
        hsi_60d_return=d.get("hsi_60d_return"),
        hsi_60d_vol_annualized=d.get("hsi_60d_vol_annualized"),
        hsi_60d_vol_pct_rank=d.get("hsi_60d_vol_pct_rank"),
        hsi_valuation_pct=d.get("hsi_valuation_pct"),
        hk_ipo_30d_avg_d30=d.get("hk_ipo_30d_avg_d30"),
        hk_ipo_30d_breakage_rate=d.get("hk_ipo_30d_breakage_rate"),
        southbound_30d_net_normalized=d.get("southbound_30d_net_normalized"),
        sector_60d_vol_annualized=d.get("sector_60d_vol_annualized"),
    )


def render_market_env_md(env: HKMarketEnv) -> str:
    """把 HKMarketEnv 渲染成 Markdown brief 块, 喂给 macro / decision agent."""
    def fmt(v: float | None, pct: bool = False, sign: bool = False) -> str:
        if v is None:
            return "—"
        if pct:
            return f"{v * 100:+.1f}%" if sign else f"{v * 100:.1f}%"
        return f"{v:+.2f}" if sign else f"{v:.2f}"

    lines = [
        f"### hkquant 市场环境快照 (asof {env.asof_month})",
        "",
        f"- HSI 60 日收益: {fmt(env.hsi_60d_return, pct=True, sign=True)}",
        f"- HSI 60 日年化波动: {fmt(env.hsi_60d_vol_annualized, pct=True)}",
        f"- HSI 波动历史百分位: {fmt(env.hsi_60d_vol_pct_rank, pct=True)} "
        f"({'高波动 ⚠' if (env.hsi_60d_vol_pct_rank or 0) > 0.7 else '常态'})",
        f"- HSI 估值百分位: {fmt(env.hsi_valuation_pct, pct=True)}",
        f"- 港股近 30 日 IPO 平均 D30 回报: "
        f"{fmt(env.hk_ipo_30d_avg_d30, pct=True, sign=True)}",
        f"- 港股近 30 日 IPO **破发率**: "
        f"{fmt(env.hk_ipo_30d_breakage_rate, pct=True)} "
        f"⭐ regime gate",
        f"- 南下 30 日净流入 (标准化): "
        f"{fmt(env.southbound_30d_net_normalized, sign=True)}",
        f"- 同板块 60 日年化波动: "
        f"{fmt(env.sector_60d_vol_annualized, pct=True)}",
    ]
    return "\n".join(lines)


# ============================================================================
# 查询: 单标的 lookup (T4 自动 outcome 用)
# ============================================================================

def get_ipo_master_by_stock_code(
    stock_code: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """按 stock_code 严格查 ipo_master 单行; 未命中返 {}.

    严格匹配 (不做模糊) 是为了 T4 自动 outcome — 不能把 A 公司的回报错算给 B.
    多家同 stock_code (理论上不应发生) 取最早 listing_date.
    """
    if not is_available() and db_path is None:
        return {}
    if not stock_code:
        return {}
    sql = (
        "SELECT * FROM ipo_master WHERE stock_code = ? "
        "ORDER BY listing_date ASC LIMIT 1"
    )
    try:
        with _connect(db_path) as conn:
            row = conn.execute(sql, (stock_code,)).fetchone()
            return dict(row) if row else {}
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_ipo_master_by_stock_code 失败: {e}")
        return {}


def get_ipo_returns_by_ipo_id(
    ipo_id: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """按 ipo_id 查 ipo_returns 单行; 未命中返 {}."""
    if not is_available() and db_path is None:
        return {}
    if not ipo_id:
        return {}
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM ipo_returns WHERE ipo_id = ?", (ipo_id,),
            ).fetchone()
            return dict(row) if row else {}
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.warning(f"[hkquant] get_ipo_returns_by_ipo_id 失败: {e}")
        return {}


# ============================================================================
# helpers
# ============================================================================

def _to_date_str(v: date | str | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v.isoformat()
    return str(v)
