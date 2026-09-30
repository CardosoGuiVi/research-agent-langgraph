"""Test doubles shared across the suite."""

from __future__ import annotations

from typing import Any

from research_agent.config import Settings
from research_agent.container import Container, build_container


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"anthropic_api_key": "sk-test", "tavily_api_key": None}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def build_test_container(**settings_overrides: Any) -> Container:
    return build_container(make_settings(**settings_overrides))
