"""主题分类器 — 把港股 IPO 排队公司归到 industry_theme 枚举.

设计:
- iFinD 给的港股排队队列含 (company_name, business_scope, hkex_industry_raw),
  但 hkex_industry 用的是港交所自己的分类 (例: "工业品") 不命中我们的
  ListingProfile.industry_theme 12 个枚举.
- 用 LLM 逐家判定, 输出 industry_theme + confidence + rationale.
- SQLite 24h 缓存: 同一家公司一天内不重跑, 控制成本 (50-100 家首次约 1-2 USD).
- LLM 判定失败时 fallback "Other".

使用:
    classifier = ThemeClassifier(llm, store)
    theme = classifier.classify(QueuedCompany(...))  # 命中缓存 = 0 token
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from src.data.ifind_ipo_queue import QueuedCompany
from src.feedback.store import FeedbackStore
from src.llm import ModelTier

INDUSTRY_THEMES = (
    "Tech_AI_Semi", "Bio_Pharma", "Med_Device", "Robotics_Automation",
    "New_Energy", "Advanced_Materials", "Consumer", "Financial",
    "Real_Estate", "Industrial", "Healthcare", "Other",
)

CLASSIFY_SYSTEM = """你是港股行业主题分类器, 把一家公司归类到下面 12 个 industry_theme 之一:

- Tech_AI_Semi: AI / 半导体 / 通用科技 (软件 / 云 / 大模型 / 芯片设计)
- Bio_Pharma: 生物医药 (创新药 / 临床期生物科技 / 18A 主战场)
- Med_Device: 医疗器械 (诊断 / 耗材 / 影像设备)
- Robotics_Automation: 机器人 / 工业自动化 (协作机器人 / 具身智能 / 工业机器人本体)
- New_Energy: 新能源 (锂电 / 储能 / 光伏 / 氢能 / 新能源车产业链)
- Advanced_Materials: 先进材料 (特种化学品 / 复合材料 / 半导体材料)
- Consumer: 消费 (品牌 / 餐饮 / 零售 / 互联网消费 / 电商)
- Financial: 金融 (银行 / 保险 / 证券 / Fintech / 资管)
- Real_Estate: 地产 (开发 / 物管 / REITs / 商业地产)
- Industrial: 工业 / 传统制造 (高端装备 / 重工业 / 化工 — 不属于上述新兴主题的)
- Healthcare: 医疗服务 (医院 / 体检 / CRO / 互联网医疗 — 非药)
- Other: 完全不属于上述任何一类

输出严格 JSON (用 ```json``` 代码块):
```json
{
  "industry_theme": "<12 个枚举之一>",
  "confidence": "高|中|低",
  "rationale": "<1 句话, 引用业务描述里的关键词>"
}
```

判定原则:
- 主业占比 > 50% 的才算; 如果是控股集团多业务, 选最大占比的.
- 业务描述里有明确技术词汇 (如"基因测序" → Bio_Pharma; "协作机器人" → Robotics_Automation) 优先按词汇判.
- 模糊时降 confidence 到 "中" 或 "低", 不要瞎猜.
- 完全无法判定就给 "Other" + confidence "低".
"""


class ThemeClassifier:
    """LLM-based industry_theme 分类器, 带 24h SQLite 缓存."""

    def __init__(self, llm: Any, store: FeedbackStore | None = None,
                 ttl_hours: int = 24):
        self.llm = llm
        self.store = store
        self.ttl_hours = ttl_hours
        self.tier = ModelTier.SUMMARIZE  # 分类只用 cheap tier

    @staticmethod
    def _normalize_key(company: QueuedCompany) -> str:
        """规范化公司名作为缓存 key."""
        if company.hk_code:
            return f"hk:{company.hk_code}"
        return f"name:{company.company_name.strip().lower()}"

    def classify(self, company: QueuedCompany) -> dict[str, Any]:
        """分类一家公司. 命中缓存 = 0 token; 未命中调 LLM 并落库."""
        key = self._normalize_key(company)

        # 1. 缓存命中?
        if self.store is not None:
            cached = self.store.get_theme_classification(
                key, ttl_hours=self.ttl_hours,
            )
            if cached is not None:
                return {
                    "industry_theme": cached["industry_theme"],
                    "confidence": cached["confidence"],
                    "rationale": cached["rationale"],
                    "from_cache": True,
                }

        # 2. 调 LLM
        try:
            theme, conf, rat = self._call_llm(company)
        except Exception as e:
            logger.warning(f"[ThemeClassifier] LLM 失败 ({company.company_name}): {e}")
            theme, conf, rat = "Other", "低", f"LLM 失败: {type(e).__name__}"

        # 3. 落缓存
        if self.store is not None:
            try:
                self.store.save_theme_classification(
                    key, industry_theme=theme,
                    confidence=conf, rationale=rat,
                )
            except Exception as e:
                logger.debug(f"[ThemeClassifier] 缓存写入失败: {e}")

        return {
            "industry_theme": theme, "confidence": conf,
            "rationale": rat, "from_cache": False,
        }

    def classify_batch(
        self, companies: list[QueuedCompany],
    ) -> list[dict[str, Any]]:
        """批量分类, 返回与输入等长的结果列表."""
        return [self.classify(c) for c in companies]

    def _call_llm(self, company: QueuedCompany) -> tuple[str, str, str]:
        """实际调 LLM. 返回 (theme, confidence, rationale)."""
        user_msg = (
            f"公司名称: {company.company_name}\n"
            f"业务描述: {company.business_scope or '(未提供)'}\n"
            f"港交所行业: {company.hkex_industry or '(未提供)'}\n"
            f"保荐人: {company.sponsor or '(未提供)'}\n\n"
            f"请输出 industry_theme 分类 JSON."
        )
        resp = self.llm.complete(
            tier=self.tier,
            system=CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=300,
            temperature=0.0,
        )
        text = resp.text if hasattr(resp, "text") else str(resp)
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            # 退化: 整段尝试 parse
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return "Other", "低", "LLM 输出未含 ```json``` 代码块"
        else:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError as e:
                return "Other", "低", f"JSON parse failed: {e}"

        theme = str(parsed.get("industry_theme", "Other"))
        if theme not in INDUSTRY_THEMES:
            return "Other", "低", f"未识别 theme '{theme}', 退化为 Other"
        conf = str(parsed.get("confidence", "中"))
        if conf not in ("高", "中", "低"):
            conf = "中"
        rat = str(parsed.get("rationale", ""))[:300]
        return theme, conf, rat
