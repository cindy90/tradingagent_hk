"""行业港股候选池构造器（peers v2）。

设计: 不让 LLM 凭训练记忆瞎编 ticker, 而是先从 akshare/iFind 拉一份"行业相关
港股池", 再让 LLM 从池子里选 + 给业务相似度打分。

数据源优先级:
1. akshare `stock_hk_spot_em` (免费, 含名称 + 市值 + 行情) → 名称关键词过滤
2. (可选) iFind THS_BD 批量补 `ths_main_business_stock` 字段做二次精筛
3. 都不可用时返回空 list, PeerSuggester 自动降级回纯招股书 RAG 提取

输出字段统一为 5 位 ticker + 中文名 + 市值（亿 HKD）+ 涨跌等基础行情，
方便 LLM 直观判断"哪几家最像 target"。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from .cache import disk_cache


# 港股市值字段在 akshare 不同版本里命名可能不同, 兜底列表
_MARKET_CAP_KEYS = ("总市值", "市值", "总市值(港元)", "总股本市值")
_TURNOVER_KEYS = ("成交额", "成交额(港元)")
_PRICE_KEYS = ("最新价", "现价")
_CHANGE_KEYS = ("涨跌幅", "涨跌幅(%)")


def _pick(row: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, "", "-"):
            return row[k]
    return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_ticker(raw: Any) -> str | None:
    """akshare 港股代码可能是 4/5 位字符串, 统一成 5 位补 0."""
    if raw is None:
        return None
    s = str(raw).strip().replace(".HK", "").replace(".hk", "")
    if not s.isdigit():
        return None
    return s.zfill(5)


@disk_cache(ttl_seconds=12 * 3600, namespace="peer_pool")
def fetch_hk_market_snapshot() -> list[dict]:
    """从 akshare 拉港股全市场快照。缓存 12 小时。

    Returns: [{ticker, name, latest_price, change_pct, market_cap_hkd_b,
              turnover_hkd_b}, ...]
    若 akshare 不可用 / 调用失败, 返回空 list。
    """
    try:
        import akshare as ak
    except ImportError:
        logger.info("akshare 未安装, 港股池构建跳过")
        return []
    try:
        df = ak.stock_hk_spot_em()
    except Exception as e:
        logger.warning(f"stock_hk_spot_em 调用失败: {e}")
        return []
    if df is None or df.empty:
        return []

    out = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ticker = _normalize_ticker(d.get("代码"))
        if not ticker:
            continue
        name = str(d.get("名称", "")).strip()
        if not name:
            continue
        market_cap_raw = _to_float(_pick(d, _MARKET_CAP_KEYS))
        # akshare 通常是港元单位（绝对值），转亿 HKD
        market_cap_b = round(market_cap_raw / 1e8, 2) if market_cap_raw else None
        turnover_raw = _to_float(_pick(d, _TURNOVER_KEYS))
        turnover_b = round(turnover_raw / 1e8, 2) if turnover_raw else None

        out.append({
            "ticker": ticker,
            "name": name,
            "latest_price": _to_float(_pick(d, _PRICE_KEYS)),
            "change_pct": _to_float(_pick(d, _CHANGE_KEYS)),
            "market_cap_hkd_b": market_cap_b,
            "turnover_hkd_b": turnover_b,
        })
    return out


def _filter_by_keywords(
    snapshot: list[dict],
    keywords: list[str],
    *,
    field: str = "name",
) -> list[dict]:
    """名称包含任一关键词即留。"""
    if not keywords:
        return list(snapshot)
    matched = []
    for r in snapshot:
        text = str(r.get(field, "")).lower()
        if any(kw.lower() in text for kw in keywords if kw):
            matched.append(r)
    return matched


def _filter_by_market_cap(
    rows: list[dict],
    min_cap_hkd_b: float | None,
    max_cap_hkd_b: float | None,
) -> list[dict]:
    """按市值范围过滤（亿 HKD），缺市值的项 **保留** (避免误杀新上市)。"""
    if min_cap_hkd_b is None and max_cap_hkd_b is None:
        return rows
    out = []
    for r in rows:
        mc = r.get("market_cap_hkd_b")
        if mc is None:
            out.append(r)  # 保留
            continue
        if min_cap_hkd_b is not None and mc < min_cap_hkd_b:
            continue
        if max_cap_hkd_b is not None and mc > max_cap_hkd_b:
            continue
        out.append(r)
    return out


def _enrich_with_business_summary(rows: list[dict], max_enrich: int = 30) -> list[dict]:
    """对前 max_enrich 个候选, 调 iFind THS_BD 补 ths_main_business_stock 字段.

    主营业务文本可大幅提升 LLM 判断业务相似度的准确性。
    iFind 未配置时静默跳过。
    """
    if not rows:
        return rows
    try:
        from src.data.ifind_sdk import get_basic_data, to_ths_hk_code
    except ImportError:
        return rows

    enriched = list(rows)
    for r in enriched[:max_enrich]:
        if "main_business" in r:
            continue
        try:
            thscode = to_ths_hk_code(r["ticker"])
            data = get_basic_data(thscode, "ths_main_business_stock", "")
            biz = (data.get("ths_main_business_stock") or "").strip()
            if biz:
                # 截断到 200 字, 给 prompt 留余量
                r["main_business"] = biz[:200]
        except Exception:
            continue
    return enriched


def build_peer_pool(
    keywords: list[str],
    *,
    target_market_cap_hkd_b: float | None = None,
    cap_lower_factor: float = 0.2,  # 0.2x ~ 5x 区间默认
    cap_upper_factor: float = 5.0,
    explicit_cap_range: tuple[float | None, float | None] | None = None,
    limit: int = 60,
    enrich_with_business: bool = True,
) -> list[dict]:
    """构造行业港股候选池。

    Args:
        keywords: 行业关键词列表（任一命中公司名即留）
        target_market_cap_hkd_b: 目标公司估值（亿 HKD）, None 表示不按规模过滤
        cap_lower_factor / cap_upper_factor: 自动按 target 估值的倍数区间过滤
            （默认 0.2x ~ 5x; 即 80 亿目标 → 16-400 亿候选）
        explicit_cap_range: 显式给定市值区间 (min, max), 优先级高于 factor
        limit: 返回最多多少家
        enrich_with_business: 是否对前 N 家调 iFind 补主营业务（精筛但慢）

    Returns: 候选 peer 列表（按市值降序），失败返回 []
    """
    snapshot = fetch_hk_market_snapshot()
    if not snapshot:
        logger.warning("港股快照为空, peer pool 无法构建（akshare 不可用）")
        return []

    logger.info(f"港股全市场快照: {len(snapshot)} 家")

    # 关键词过滤
    filtered = _filter_by_keywords(snapshot, keywords)
    logger.info(f"关键词 {keywords} 过滤后: {len(filtered)} 家")

    # 市值范围
    if explicit_cap_range:
        min_cap, max_cap = explicit_cap_range
    elif target_market_cap_hkd_b:
        min_cap = target_market_cap_hkd_b * cap_lower_factor
        max_cap = target_market_cap_hkd_b * cap_upper_factor
    else:
        min_cap, max_cap = None, None
    if min_cap or max_cap:
        filtered = _filter_by_market_cap(filtered, min_cap, max_cap)
        logger.info(f"市值范围 [{min_cap}, {max_cap}] 过滤后: {len(filtered)} 家")

    # 按市值降序排（None 沉底）
    filtered.sort(
        key=lambda r: r.get("market_cap_hkd_b") or 0,
        reverse=True,
    )
    filtered = filtered[:limit]

    # iFind 主营业务精筛
    if enrich_with_business and filtered:
        filtered = _enrich_with_business_summary(filtered)

    return filtered


def derive_keywords_from_industry(industry: str) -> list[str]:
    """从 industry 字段推断关键词列表（例:"工业机器人/协作机器人" → ["机器人","工业","协作"]）。

    简单规则: 按 / 和 空格 拆, 取每段长度 ≥ 2 的子串。
    """
    if not industry:
        return []
    raw_parts = [p.strip() for p in industry.replace("、", "/").replace(",", "/").split("/")]
    out: set[str] = set()
    for p in raw_parts:
        if not p:
            continue
        out.add(p)
        # 进一步按空格拆
        for sub in p.split():
            if len(sub) >= 2:
                out.add(sub)
    return [w for w in out if len(w) >= 2]
