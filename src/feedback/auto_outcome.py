"""T4: 自动从 hkquant 读 ipo_returns → 给本项目 closed/open prediction 落 Outcome.

闭环全自动化:
- T2.5 export 是 tradingagent → hkquant 写入 (反哺训练集)
- T4 是 hkquant → tradingagent 读取 (用真实 ipo_returns 自动评 outcome)

主流程:
1. 扫 FeedbackStore 所有 prediction (除 status='dead')
2. 对每个 prediction: 用 ticker 严格匹配 hkquant.ipo_master (no fuzzy)
   命中后从 hkquant.ipo_returns 拼 Outcome dataclass
3. 与现有 latest_outcome 比较: hkquant 数据更新 (有 m6/m12 而 outcome 没) 时
   才落库; 否则跳过
4. 可选: hkquant 有 ≥ M6 数据 → 自动 update_status('closed') (--auto-close)

安全保证:
- 默认 dry-run, --apply 才实际写
- 严格 stock_code 匹配 (不模糊)
- 只补充 / 覆盖, 不删除已有 outcome 字段
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from loguru import logger

from src.data import hkquant_client
from src.feedback.models import Outcome, Prediction
from src.feedback.store import FeedbackStore


# ============================================================================
# 字段映射纯函数 (易于单测)
# ============================================================================

def _to_date(v: Any) -> date | None:
    """ISO 字符串 / date / datetime → date; 不可解析返 None."""
    if v is None or v == "":
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.fromisoformat(str(v)).date()
    except (ValueError, TypeError):
        return None


def derive_was_broken_ipo(
    d1: float | None, d30: float | None, d180: float | None,
) -> tuple[bool | None, bool | None]:
    """从已知 horizon 回报派生破发标记.

    Returns:
        (was_broken_ipo_d1, was_broken_ipo_d180)
        d1 / d180 任一为 None 时对应输出 None (避免假阳性)
    """
    broken_d1 = (d1 < 0) if d1 is not None else None
    # 180 日内"曾经破发": 取已知 horizon 中是否有 < 0
    horizons = [v for v in (d1, d30, d180) if v is not None]
    broken_d180 = None
    if d180 is not None:
        broken_d180 = any(h < 0 for h in horizons)
    elif d30 is not None:
        # 没 d180 但有 d30, 给一个保守估计
        broken_d180 = any(h < 0 for h in horizons)
    return broken_d1, broken_d180


def build_outcome_from_hkquant(
    prediction_id: int,
    master: dict[str, Any],
    returns: dict[str, Any],
    *,
    recorded_date: datetime | None = None,
) -> Outcome | None:
    """把 hkquant ipo_master + ipo_returns 拼成 Outcome.

    master 必须含 listing_date (否则 Outcome 无意义) 才返回; 缺则返 None.
    returns 可以全空 (会变成只有 listing_date + ipo_actual_price 的 Outcome).
    """
    if not master:
        return None
    listing_date = _to_date(master.get("listing_date"))
    if listing_date is None:
        return None

    d1 = returns.get("return_d1_close") if returns else None
    d30 = returns.get("return_d30") if returns else None
    d180 = returns.get("return_m6") if returns else None       # m6 ≈ d180
    d365 = returns.get("return_m12") if returns else None      # m12 ≈ d365

    broken_d1, broken_d180 = derive_was_broken_ipo(d1, d30, d180)

    avg_turnover_m_hkd = None
    avg_vol_hkd = returns.get("avg_daily_volume_hkd") if returns else None
    if avg_vol_hkd is not None:
        avg_turnover_m_hkd = float(avg_vol_hkd) / 1e6

    return Outcome(
        prediction_id=prediction_id,
        recorded_date=recorded_date or datetime.utcnow(),
        ipo_actual_price_hkd=master.get("offer_price_hkd"),
        final_listing_date=listing_date,
        d1_return=d1,
        d30_return=d30,
        d90_return=None,                                  # hkquant 无 m3
        d180_return=d180,
        d365_return=d365,
        was_broken_ipo_d1=broken_d1,
        was_broken_ipo_d180=broken_d180,
        max_drawdown_in_lockup_pct=(
            returns.get("max_drawdown_m6") if returns else None
        ),
        avg_daily_turnover_hkd_m_d180=avg_turnover_m_hkd,
        user_notes="auto_outcome from hkquant",
        notable_events=[],
    )


def is_outcome_more_complete(new_o: Outcome, existing: Outcome | None) -> bool:
    """判断新 outcome 是否比现有更完整 (hkquant 拿到了之前没有的 horizon)."""
    if existing is None:
        return True
    # 关键字段优先级: m12 > m6 > d30 > d1
    for field in ("d365_return", "d180_return", "d30_return", "d1_return"):
        new_v = getattr(new_o, field, None)
        old_v = getattr(existing, field, None)
        if new_v is not None and old_v is None:
            return True
    return False


# ============================================================================
# 主流程
# ============================================================================

@dataclass
class AutoOutcomeAction:
    """单个 prediction 的 auto-outcome 决策结果."""
    prediction_id: int
    project_id: str
    ticker: str
    matched_in_hkquant: bool
    would_write_outcome: bool        # dry-run 时仅描述
    would_close_status: bool
    skip_reason: str = ""
    new_outcome: Outcome | None = None


@dataclass
class BatchStats:
    scanned: int
    matched: int
    written: int
    skipped_no_match: int
    skipped_no_new_data: int
    closed: int


def auto_record_outcome_for_prediction(
    prediction: Prediction,
    *,
    store: FeedbackStore,
    db_path=None,
    dry_run: bool = True,
    auto_close: bool = False,
) -> AutoOutcomeAction:
    """对单个 prediction 自动 score outcome.

    Args:
        prediction: 已落库的 Prediction 对象
        store: FeedbackStore (供查 latest_outcome / 写 record_outcome)
        db_path: hkquant DB 注入 (测试用; 生产读 settings)
        dry_run: True (默认) 不写库, 只返 Action
        auto_close: True 且新 outcome 有 ≥ M6 数据时, 自动 status → 'closed'
    """
    # 1) 找 prediction 的 SQL int id
    row = store._conn.execute(
        "SELECT id, status FROM predictions WHERE project_id = ?",
        (prediction.project_id,),
    ).fetchone()
    if row is None:
        return AutoOutcomeAction(
            prediction_id=-1, project_id=prediction.project_id,
            ticker=prediction.ticker, matched_in_hkquant=False,
            would_write_outcome=False, would_close_status=False,
            skip_reason="prediction 在 store 中找不到",
        )
    pid = int(row["id"])
    cur_status = row["status"]

    # 2) 严格 stock_code 匹配 hkquant.ipo_master
    master = hkquant_client.get_ipo_master_by_stock_code(
        prediction.ticker, db_path=db_path,
    )
    if not master:
        return AutoOutcomeAction(
            prediction_id=pid, project_id=prediction.project_id,
            ticker=prediction.ticker, matched_in_hkquant=False,
            would_write_outcome=False, would_close_status=False,
            skip_reason=f"hkquant.ipo_master 中无 stock_code='{prediction.ticker}'",
        )

    # 3) 拉 returns (可能为空, 拼 Outcome 不强制要求)
    returns = hkquant_client.get_ipo_returns_by_ipo_id(
        master.get("ipo_id", ""), db_path=db_path,
    )
    new_o = build_outcome_from_hkquant(pid, master, returns)
    if new_o is None:
        return AutoOutcomeAction(
            prediction_id=pid, project_id=prediction.project_id,
            ticker=prediction.ticker, matched_in_hkquant=True,
            would_write_outcome=False, would_close_status=False,
            skip_reason="hkquant 命中但 listing_date 缺失, 无法构造 Outcome",
        )

    # 4) 比较现有 outcome — 没新数据就跳
    existing = store.latest_outcome(pid)
    if not is_outcome_more_complete(new_o, existing):
        return AutoOutcomeAction(
            prediction_id=pid, project_id=prediction.project_id,
            ticker=prediction.ticker, matched_in_hkquant=True,
            would_write_outcome=False, would_close_status=False,
            skip_reason="hkquant 数据无新增 horizon, 跳过",
            new_outcome=new_o,
        )

    # 5) 决定是否 auto-close
    has_m6 = new_o.d180_return is not None
    will_close = bool(auto_close and has_m6 and cur_status != "closed")

    action = AutoOutcomeAction(
        prediction_id=pid, project_id=prediction.project_id,
        ticker=prediction.ticker, matched_in_hkquant=True,
        would_write_outcome=True, would_close_status=will_close,
        new_outcome=new_o,
    )
    if dry_run:
        return action

    # 6) 实际写库
    store.record_outcome(new_o)
    if will_close:
        store.update_status(pid, "closed")
    return action


def run_batch(
    *,
    store: FeedbackStore | None = None,
    db_path=None,
    dry_run: bool = True,
    auto_close: bool = False,
    only_status: str | None = None,
    only_ticker: str | None = None,
) -> tuple[BatchStats, list[AutoOutcomeAction]]:
    """扫所有 prediction, 对每个跑 auto_record_outcome_for_prediction.

    Args:
        only_status: 仅扫指定 status (例 'open'); None 全扫
        only_ticker: 仅一个 ticker (CLI --ticker)
    """
    own = store is None
    s = store or FeedbackStore()
    try:
        preds = s.list_predictions(
            status=only_status, ticker=only_ticker, limit=10000,
        )
        actions: list[AutoOutcomeAction] = []
        n_match = n_write = n_close = 0
        n_no_match = n_no_new = 0
        for p in preds:
            a = auto_record_outcome_for_prediction(
                p, store=s, db_path=db_path,
                dry_run=dry_run, auto_close=auto_close,
            )
            actions.append(a)
            if not a.matched_in_hkquant:
                n_no_match += 1
                continue
            n_match += 1
            if not a.would_write_outcome:
                n_no_new += 1
                continue
            n_write += 1
            if a.would_close_status:
                n_close += 1
        stats = BatchStats(
            scanned=len(preds), matched=n_match, written=n_write,
            skipped_no_match=n_no_match, skipped_no_new_data=n_no_new,
            closed=n_close,
        )
        logger.info(
            f"[auto_outcome] 扫 {stats.scanned} | hkquant 命中 {stats.matched} | "
            f"{'拟' if dry_run else '已'}写 {stats.written} | "
            f"未命中 {stats.skipped_no_match} | 无新数据 {stats.skipped_no_new_data} | "
            f"close {stats.closed}"
        )
        return stats, actions
    finally:
        if own:
            s.close()


# ============================================================================
# CLI
# ============================================================================

def _format_action(a: AutoOutcomeAction) -> str:
    if not a.matched_in_hkquant:
        return f"  [SKIP] {a.ticker:<10} ({a.project_id}): {a.skip_reason}"
    if not a.would_write_outcome:
        return f"  [NOOP] {a.ticker:<10} ({a.project_id}): {a.skip_reason}"
    o = a.new_outcome
    horizons = "/".join(
        f"{name}={getattr(o, attr)*100:+.1f}%" if getattr(o, attr) is not None else f"{name}=—"
        for name, attr in [("D1", "d1_return"), ("D30", "d30_return"),
                           ("M6", "d180_return"), ("M12", "d365_return")]
    )
    close_tag = " → close" if a.would_close_status else ""
    return f"  [WRITE] {a.ticker:<10} ({a.project_id}): {horizons}{close_tag}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="auto_outcome",
        description=(
            "用 hkquant.ipo_returns 自动给本项目 prediction 落 Outcome. "
            "默认 dry-run, --apply 才实际写库."
        ),
    )
    ap.add_argument("--apply", action="store_true",
                    help="实际写库 (默认 dry-run)")
    ap.add_argument("--auto-close", action="store_true",
                    help="新 outcome 有 ≥ M6 数据时自动 status → 'closed'")
    ap.add_argument("--status", default=None,
                    help="仅扫指定 status (例 'open' / 'closed')")
    ap.add_argument("--ticker", default=None,
                    help="仅扫一个 ticker (例 9999.HK)")
    args = ap.parse_args(argv)

    if not hkquant_client.is_available():
        print(
            "[ERROR] HKQUANT_DB_PATH 未配置或文件不存在. "
            "无法跑自动 outcome (T4 依赖 hkquant 数据).",
            file=sys.stderr,
        )
        return 2

    dry_run = not args.apply
    stats, actions = run_batch(
        dry_run=dry_run, auto_close=args.auto_close,
        only_status=args.status, only_ticker=args.ticker,
    )

    print(f"\n=== auto_outcome ({'DRY RUN' if dry_run else 'APPLY'}) ===\n")
    for a in actions:
        print(_format_action(a))
    print(
        f"\n汇总: 扫 {stats.scanned} | 命中 {stats.matched} | "
        f"{'拟' if dry_run else '已'}写 {stats.written} | "
        f"未命中 {stats.skipped_no_match} | 无新数据 {stats.skipped_no_new_data} | "
        f"close {stats.closed}"
    )
    if dry_run and stats.written > 0:
        print("\n要实际写库请加 --apply (可选 --auto-close 自动迁移 status).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
