"""Application settings, loaded from environment variables (and `.env` in local dev)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class SearchProviderName(StrEnum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Secrets (never logged; SecretStr hides them from repr/str) ---
    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # --- LLM ---
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    llm_model_fast: str = "claude-haiku-4-5"
    llm_model_smart: str = "claude-opus-5"
    llm_smart_effort: Effort = "medium"
    # OpenRouter (used when LLM_PROVIDER=openrouter); ids look like "vendor/model".
    openrouter_model: str | None = None
    openrouter_model_smart: str | None = None  # final answer; defaults to openrouter_model
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    max_tokens_planner: int = Field(default=2_000, gt=0)
    max_tokens_research: int = Field(default=2_000, gt=0)
    max_tokens_analysis: int = Field(default=2_000, gt=0)
    max_tokens_answer: int = Field(default=16_000, gt=0)
    max_tokens_per_run: int = Field(default=400_000, gt=0)

    # --- Graph limits ---
    max_iterations: int = Field(default=2, ge=1, le=5)
    max_research_steps: int = Field(default=3, ge=1, le=10)
    max_searches_per_run: int = Field(default=20, ge=1)
    max_fetches_per_run: int = Field(default=20, ge=0)
    min_sub_questions: int = Field(default=3, ge=1)
    max_sub_questions: int = Field(default=6, ge=1)

    # --- Tools ---
    search_results_per_query: int = Field(default=5, ge=1, le=10)
    search_timeout_s: float = Field(default=15.0, gt=0)
    fetch_timeout_s: float = Field(default=15.0, gt=0)
    fetch_max_bytes: int = Field(default=2_000_000, gt=0)
    fetch_max_chars: int = Field(default=6_000, gt=0)
    tool_max_attempts: int = Field(default=3, ge=1)

    # --- Service ---
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def search_provider(self) -> SearchProviderName:
        """Tavily when a key is configured, otherwise the keyless DuckDuckGo fallback."""
        if self.tavily_api_key and self.tavily_api_key.get_secret_value().strip():
            return SearchProviderName.TAVILY
        return SearchProviderName.DUCKDUCKGO


@lru_cache
def get_settings() -> Settings:
    return Settings()
