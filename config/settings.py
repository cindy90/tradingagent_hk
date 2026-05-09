from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    kimi_api_key: str = Field(default="", alias="KIMI_API_KEY")
    kimi_base_url: str = Field(
        default="https://api.moonshot.cn/v1",
        validation_alias=AliasChoices("KIMI_URL", "KIMI_BASE_URL"),
    )
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")

    ths_refresh_token: str = Field(
        default="",
        validation_alias=AliasChoices("THS_REFRESH_TOKEN", "IFIND_REFRESH_TOKEN"),
    )
    ths_token_cache: str = Field(default="", alias="THS_TOKEN_CACHE")
    ths_token_ttl: int = Field(default=21600, alias="THS_TOKEN_TTL")
    # iFinD Python SDK 的账号密码（不同于 REST 的 refresh_token，但是同一个 iFinD 账号）
    # SDK 用于拉 peers 财务/估值倍数/K 线/南向资金等 REST 不支持的接口
    ifind_username: str = Field(default="", alias="IFIND_USERNAME")
    ifind_password: str = Field(default="", alias="IFIND_PASSWORD")
    hkex_api_key: str = Field(default="", alias="HKEX_API_KEY")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    cache_dir: Path = Field(default=Path("./.cache"), alias="CACHE_DIR")
    reports_dir: Path = Field(default=Path("./reports"), alias="REPORTS_DIR")
    prospectus_dir: Path = Field(default=Path("./data/prospectus"), alias="PROSPECTUS_DIR")

    model_tier_summarize: str = Field(default="", alias="MODEL_TIER_SUMMARIZE")
    model_tier_analyze: str = Field(default="", alias="MODEL_TIER_ANALYZE")
    model_tier_decide: str = Field(default="", alias="MODEL_TIER_DECIDE")

    # Provider-specific tier overrides
    model_tier_summarize_anthropic: str = Field(default="", alias="MODEL_TIER_SUMMARIZE_ANTHROPIC")
    model_tier_analyze_anthropic: str = Field(default="", alias="MODEL_TIER_ANALYZE_ANTHROPIC")
    model_tier_decide_anthropic: str = Field(default="", alias="MODEL_TIER_DECIDE_ANTHROPIC")
    model_tier_summarize_kimi: str = Field(default="", alias="MODEL_TIER_SUMMARIZE_KIMI")
    model_tier_analyze_kimi: str = Field(default="", alias="MODEL_TIER_ANALYZE_KIMI")
    model_tier_decide_kimi: str = Field(default="", alias="MODEL_TIER_DECIDE_KIMI")
    model_tier_summarize_deepseek: str = Field(default="", alias="MODEL_TIER_SUMMARIZE_DEEPSEEK")
    model_tier_analyze_deepseek: str = Field(default="", alias="MODEL_TIER_ANALYZE_DEEPSEEK")
    model_tier_decide_deepseek: str = Field(default="", alias="MODEL_TIER_DECIDE_DEEPSEEK")

    debate_max_rounds: int = Field(default=2, alias="DEBATE_MAX_ROUNDS")
    rag_top_k: int = Field(default=6, alias="RAG_TOP_K")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    embedding_model: str = Field(
        default="BAAI/bge-base-zh-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    ths_endpoint_basic_data: str = Field(
        default="basic_data_service", alias="THS_ENDPOINT_BASIC_DATA"
    )
    ths_endpoint_edb: str = Field(default="edb_service", alias="THS_ENDPOINT_EDB")
    ths_endpoint_data_report: str = Field(
        default="data_report", alias="THS_ENDPOINT_DATA_REPORT"
    )

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.reports_dir, self.prospectus_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
