from enum import Enum

from config import get_settings


class ModelTier(str, Enum):
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    DECIDE = "decide"


def resolve_model(tier: ModelTier) -> str:
    s = get_settings()
    return {
        ModelTier.SUMMARIZE: s.model_tier_summarize,
        ModelTier.ANALYZE: s.model_tier_analyze,
        ModelTier.DECIDE: s.model_tier_decide,
    }[tier]
