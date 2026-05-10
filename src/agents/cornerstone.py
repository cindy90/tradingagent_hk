"""CornerstoneAgent (T2): 基石投资人质量分析.

流程:
1. 读 ctx.briefs["prospectus_analyst"] 中的"七、基石投资条款解读"段落
   (依赖 ProspectusAnalystAgent 已经跑过)
2. 用 cheap LLM tier (Haiku/Kimi-8k) 把自然语言段落 → 结构化基石名称列表
3. 调 hkquant_cornerstone.enrich_cornerstone_names 解析每个名字的 ID + 历史业绩
4. 用 ANALYZE tier 综合给出"基石阵容质量"判断, 输出 CornerstoneScoreCard

降级路径:
- HKQUANT_DB_PATH 未配置 → 全部 unmatched, 仍可输出"无 hkquant 数据可参考"brief
- prospectus_analyst brief 缺失 → 抛 RuntimeError (workflow 应保证顺序)
"""
from __future__ import annotations

import json
import re
from datetime import date

from loguru import logger

from src.agents._template import TemplateAgent
from src.agents.base import AgentContext
from src.data.hkquant_cornerstone import (
    CornerstoneInfo,
    enrich_cornerstone_names,
    render_enrichment_md,
)
from src.feedback.models import CornerstoneScoreCard
from src.llm import ModelTier


SYSTEM = """你是港股 IPO 基石投资委员会的资深分析师, 专门评估**基石阵容质量**.

# 任务
基于招股书提取的基石名单 + hkquant 历史回溯数据, 判断本次基石阵容的质量,
关注 5 个维度:

1. **阵容厚度**: 命中 hkquant 的有 track record 的基石占比. < 50% 视为"陌生面孔",
   提示尽调风险 (可能是关联交易方 / 临时凑数).
2. **长线锚定**: 是否至少 1 家 hkquant 标记 is_longterm 的投资人 (主权基金 /
   养老金 / 长期 PE). 长线锚定 → 6 月解禁后抛压可控.
3. **历史业绩**: 阵容平均 M6 胜率 / M6 回报 / 锁定期纪律分. M6 胜率 < 50% +
   M6 回报为负 → 该阵容历史踩雷率高, **大幅下调质量**.
4. **资金属性**: 中资 / 外资 / 主权 / 战略性产投 比例. 全是中资 → 海外定价
   依赖度低 (双刃剑); 全是战略性产投 → 警惕业务关联 (后续股价被基本面而非
   情绪驱动, 解禁抛压更可控).
5. **板块匹配**: hkquant 给的 sector_expertise 是否匹配本项目 GICS — 不匹配
   = 投资人在陌生赛道下注, 信号弱.

# 评分标准 (cornerstone_quality_score, 1-5)
- 5: ≥ 50% 命中 + ≥ 1 长线锚定 + 阵容 M6 胜率 ≥ 60% + 板块匹配
- 4: ≥ 30% 命中 + ≥ 1 长线锚定 + M6 胜率 50-60%
- 3: ≥ 30% 命中 + 历史业绩中性
- 2: < 30% 命中 OR 阵容 M6 胜率 < 50%
- 1: 全是陌生面孔 OR 历史踩雷率高

# 输出 (严格 JSON, 在 ```json``` 代码块):

```json
{
  "summary": "<≤80 字一句话, 引用关键数字>",
  "overall_score": <1-5, 同 cornerstone_quality_score>,
  "cornerstone_quality_score": <1-5>,
  "confidence": "高|中|低",
  "extracted_count": <int>,
  "matched_count": <int>,
  "has_longterm_anchor": <true/false>,
  "chinese_capital_pct": <0-1, 中资命中数 / 命中数>,
  "avg_winrate_m6_5y": <0-1 或 null>,
  "avg_lockup_discipline": <0-1 或 null>,
  "flagged_concerns": ["<具体问题, 必须可查>", ...],
  "notes": "<≤100 字, 说明评分依据>"
}
```

判定原则:
- 数字必须**直接引用**上方 hkquant 数据块 (不准编造).
- 命中 0 家 → score ≤ 2, confidence "低", flagged_concerns 必须包含
  "hkquant 数据缺失或全部基石未被识别".
- 不要重复 prospectus_analyst 已说过的"基石锁定期 6 个月" — 只补它没法说的
  历史业绩判断.
"""


# 提取阶段的 system prompt (Haiku/cheap tier)
EXTRACT_SYSTEM = """从招股书"基石投资条款"段落抽取**结构化基石投资人列表**.

输出严格 JSON, 用 ```json``` 代码块:

```json
{
  "cornerstones": [
    {"name": "<原文名称>", "ticket_size_hkd": <数字 or null>, "lockup_months": <数字 or null>}
  ]
}
```

规则:
- name 抄招股书原文 (含括号注释), 不要标准化. 例: "高瓴资本管理有限公司"
- ticket_size_hkd 单位 HKD (亿 HKD * 1e8); 不知道就 null
- lockup_months 默认 6 (港股惯例); 文中明示其他数字才覆盖
- 没有任何基石 → 返 {"cornerstones": []}
"""


def _safe_loads(s: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def extract_cornerstone_names_from_brief(
    llm, brief_text: str, *, max_tokens: int = 800,
) -> list[dict]:
    """跑 cheap tier LLM, 把 prospectus_analyst 的 brief 中基石段落 → 结构化列表."""
    if not brief_text:
        return []
    try:
        resp = llm.complete(
            tier=ModelTier.SUMMARIZE,
            system=EXTRACT_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    "招股书分析师简报 (相关段落):\n\n"
                    + brief_text[:8000]  # 控成本; brief 通常 << 8k
                    + "\n\n请抽取基石投资人结构化列表."
                ),
            }],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"[CornerstoneAgent] 抽取 LLM 调用失败: {e}")
        return []
    text = resp.text if hasattr(resp, "text") else str(resp)
    parsed = _safe_loads(text)
    return parsed.get("cornerstones", []) or []


class CornerstoneAgent(TemplateAgent):
    name = "cornerstone"
    description = "基石投资人质量分析 Agent (hkquant 解析)"
    tier = ModelTier.ANALYZE
    SYSTEM = SYSTEM
    score_card_class = CornerstoneScoreCard

    def build_user_message(self, ctx: AgentContext) -> str:
        # 1) 读 prospectus_analyst brief
        prospectus_brief = ctx.briefs.get("prospectus_analyst", "")
        if not prospectus_brief:
            logger.warning(
                "[CornerstoneAgent] 缺 prospectus_analyst brief, 仅做 hkquant 摘要"
            )

        # 2) cheap tier 抽取基石名称列表
        extracted = extract_cornerstone_names_from_brief(self.llm, prospectus_brief)
        names = [e.get("name", "").strip() for e in extracted if e.get("name")]
        logger.info(f"[CornerstoneAgent] 抽取 {len(names)} 家基石: {names[:5]}...")

        # 3) hkquant 解析 + 历史业绩
        asof = (
            ctx.decision_date
            if hasattr(ctx, "decision_date") and ctx.decision_date
            else date.today().isoformat()
        )
        try:
            enriched = enrich_cornerstone_names(names, asof=asof)
        except Exception as e:
            logger.warning(f"[CornerstoneAgent] hkquant 解析失败 (降级): {e}")
            enriched: list[CornerstoneInfo] = []

        # 4) 写入 ctx.extras.misc 供下游 (decision/risk) 引用
        ctx.extras.misc["cornerstone_enrichment"] = [
            {
                "raw_name": r.raw_name,
                "cornerstone_id": r.cornerstone_id,
                "confidence": r.confidence,
                "canonical_name": r.canonical_name,
                "is_longterm": r.is_longterm,
                "is_chinese": r.is_chinese,
                "winrate_m6_5y": r.winrate_m6_5y,
                "avg_m6_return_5y": r.avg_m6_return_5y,
                "lockup_discipline_score": r.lockup_discipline_score,
            }
            for r in enriched
        ]

        # 5) 渲染 prompt
        enrichment_md = render_enrichment_md(enriched)
        target_md = (
            f"# 项目\n{ctx.company_name} ({ctx.ticker})  "
            f"行业: {ctx.industry}\n"
        )
        prospectus_excerpt = (
            f"# 招股书分析师简报 (基石条款相关原文)\n"
            f"{prospectus_brief[:4000]}\n"
            if prospectus_brief
            else "# (无 prospectus_analyst 简报)\n"
        )
        return (
            f"{target_md}\n"
            f"{prospectus_excerpt}\n"
            f"{enrichment_md}\n\n"
            f"请输出基石阵容质量 JSON 评分卡. **数字必须直接引用上方 hkquant 数据**, "
            f"不准编造历史业绩."
        )
