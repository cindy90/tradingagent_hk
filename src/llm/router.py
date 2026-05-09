"""Tier → model 路由。每个 provider 有自己的默认 tier 模型映射。

ENV 优先级：
  MODEL_TIER_<TIER>_<PROVIDER> > MODEL_TIER_<TIER> > 内置默认
例如 MODEL_TIER_ANALYZE_KIMI=kimi-latest 会覆盖 Kimi provider 下的 ANALYZE tier。
"""
from __future__ import annotations

from enum import Enum

from config import get_settings


class ModelTier(str, Enum):
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    DECIDE = "decide"


# 内置默认值（按 provider × tier）
DEFAULT_MODELS: dict[str, dict[ModelTier, str]] = {
    "anthropic": {
        ModelTier.SUMMARIZE: "claude-haiku-4-5-20251001",
        ModelTier.ANALYZE: "claude-sonnet-4-6",
        ModelTier.DECIDE: "claude-opus-4-7",
    },
    "kimi": {
        # Moonshot 模型：8k/32k/128k context；kimi-latest 自动选择 context size
        # auto-context 模式按需路由，对长招股书友好
        ModelTier.SUMMARIZE: "moonshot-v1-8k",
        ModelTier.ANALYZE: "moonshot-v1-32k",
        ModelTier.DECIDE: "kimi-latest",
    },
    "deepseek": {
        # DeepSeek-V3 chat: 通用对话；deepseek-reasoner 用于决策（更强推理）
        ModelTier.SUMMARIZE: "deepseek-chat",
        ModelTier.ANALYZE: "deepseek-chat",
        ModelTier.DECIDE: "deepseek-reasoner",
    },
}


def resolve_model(tier: ModelTier) -> str:
    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower()

    # 优先级 1: provider-specific env override
    env_specific = {
        ModelTier.SUMMARIZE: f"model_tier_summarize_{provider}",
        ModelTier.ANALYZE: f"model_tier_analyze_{provider}",
        ModelTier.DECIDE: f"model_tier_decide_{provider}",
    }[tier]
    val = getattr(s, env_specific, "") or ""
    if val:
        return val

    # 优先级 2: generic env override
    val = {
        ModelTier.SUMMARIZE: s.model_tier_summarize,
        ModelTier.ANALYZE: s.model_tier_analyze,
        ModelTier.DECIDE: s.model_tier_decide,
    }[tier]
    if val and provider == "anthropic":
        return val

    # 优先级 3: 内置默认
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])[tier]
