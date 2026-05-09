from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    ths_refresh_token: str = Field(default="", alias="THS_REFRESH_TOKEN")
    ths_token_cache: str = Field(default="", alias="THS_TOKEN_CACHE")
    ths_token_ttl: int = Field(default=21600, alias="THS_TOKEN_TTL")
    hkex_api_key: str = Field(default="", alias="HKEX_API_KEY")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    cache_dir: Path = Field(default=Path("./.cache"), alias="CACHE_DIR")
    reports_dir: Path = Field(default=Path("./reports"), alias="REPORTS_DIR")
    prospectus_dir: Path = Field(default=Path("./data/prospectus"), alias="PROSPECTUS_DIR")

    model_tier_summarize: str = Field(
        default="claude-haiku-4-5-20251001", alias="MODEL_TIER_SUMMARIZE"
    )
    model_tier_analyze: str = Field(default="claude-sonnet-4-6", alias="MODEL_TIER_ANALYZE")
    model_tier_decide: str = Field(default="claude-opus-4-7", alias="MODEL_TIER_DECIDE")

    debate_max_rounds: int = Field(default=2, alias="DEBATE_MAX_ROUNDS")
    rag_top_k: int = Field(default=6, alias="RAG_TOP_K")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.reports_dir, self.prospectus_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
