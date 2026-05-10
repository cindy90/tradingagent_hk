"""HK IPO 排队公司列表抓取 — iFinD 接入点 (v2 流量稀缺度).

设计:
- iFinD 提供港股全量「申请版本/通过聆讯」队列, 但**行业分类不直接给到我们的
  industry_theme 枚举**, 需要二次分类 (见 src/agents/theme_classifier.py).
- 本模块只负责拉原始 queue, 不做主题分类.
- 24h disk_cache: queue 队列变动慢, 一天拉一次足够.

iFinD 接口 (待用户确认具体函数名 / report_id):
- 候选: THS_DR("HKEX_IPO_QUEUE", ...) 或 THS_DateSerial("...")
- 期望返回字段: company_name / hk_code / sponsor / status / submission_date /
  business_scope / hkex_industry
- 临时 stub 返回 [], 用户提供函数名后, 把 fetch_hk_ipo_queue 内部 ~5 行换掉即可
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .cache import disk_cache


@dataclass
class QueuedCompany:
    """港股 IPO 排队公司原始记录 (未分类主题)."""
    company_name: str
    hk_code: str | None = None         # 临时副牌 H 代码 (例 H1234)
    sponsor: str | None = None         # 保荐人
    status: str | None = None          # 申请版本 / 已通过聆讯 / 已撤回 / 已失效
    submission_date: str | None = None  # ISO YYYY-MM-DD
    business_scope: str = ""            # 一句话业务描述, 喂给 LLM 分类用
    hkex_industry: str | None = None    # 港交所原始行业分类 (作为 fallback hint)
    raw: dict[str, Any] = field(default_factory=dict)


@disk_cache(ttl_seconds=24 * 3600, namespace="hk_ipo_queue")
def fetch_hk_ipo_queue(
    *,
    statuses: tuple[str, ...] = ("申请版本", "已通过聆讯"),
) -> list[dict[str, Any]]:
    """从 iFinD 拉取港股 IPO 全量排队列表.

    Args:
        statuses: 关心的状态 (默认: 在审 + 已聆讯; 不含已撤回/已失效).

    Returns:
        list of dicts (用 dict 而非 dataclass 方便 disk_cache 序列化).
        失败 / 未配置 时返 [].

    TODO (用户提供 iFinD 接口名后实现):
        from .ifind_sdk import _ensure_login, get_basic_data
        # 例: data = THS_DataPool("HKEX_IPO_QUEUE", ...) 或
        #     data = THS_DR("HKEX_HK_IPO_QUEUE", "params=...")
        # parse 返回 → list[dict] 字段对齐 QueuedCompany
    """
    logger.warning(
        "[ifind_ipo_queue] fetch_hk_ipo_queue 是 stub, 返回 []. "
        "用户提供 iFinD 函数名后, 把本函数内部填充即可."
    )
    return []


def parse_queue_records(raw: list[dict[str, Any]]) -> list[QueuedCompany]:
    """把 dict 列表转成 QueuedCompany dataclass 列表.

    隔离原始 dict 形态, 让下游 (theme_classifier / scarcity_engine) 拿到
    强类型对象. 字段名宽容映射 (LLM-friendly key 都接受).
    """
    out: list[QueuedCompany] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        name = (
            r.get("company_name") or r.get("name")
            or r.get("公司名称") or r.get("申请人")
            or ""
        ).strip()
        if not name:
            continue
        out.append(QueuedCompany(
            company_name=name,
            hk_code=r.get("hk_code") or r.get("code") or r.get("ticker") or None,
            sponsor=r.get("sponsor") or r.get("保荐人") or None,
            status=r.get("status") or r.get("状态") or None,
            submission_date=(
                r.get("submission_date") or r.get("递表日期")
                or r.get("date") or None
            ),
            business_scope=(
                r.get("business_scope") or r.get("业务描述")
                or r.get("description") or ""
            ),
            hkex_industry=(
                r.get("hkex_industry") or r.get("行业") or None
            ),
            raw=r,
        ))
    return out


def get_hk_ipo_queue() -> list[QueuedCompany]:
    """便捷封装: 拉 + 解析为 QueuedCompany 列表 (供上层直接调用)."""
    return parse_queue_records(fetch_hk_ipo_queue())
