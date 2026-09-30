"""Test doubles shared across the suite."""

from __future__ import annotations

from typing import Any

from research_agent.config import Settings
from research_agent.container import Container, build_container
from research_agent.tools.search import SearchError, SearchResult


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"anthropic_api_key": "sk-test", "tavily_api_key": None}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def build_test_container(**settings_overrides: Any) -> Container:
    return build_container(make_settings(**settings_overrides))


class FakeSearchProvider:
    """Deterministic SearchProvider: canned results per query, optional scripted failures."""

    name = "fake"

    def __init__(
        self,
        responses: dict[str, list[SearchResult]] | None = None,
        *,
        default: list[SearchResult] | None = None,
        fail_times: int = 0,
        error: Exception | None = None,
    ) -> None:
        self._responses = {k.lower(): v for k, v in (responses or {}).items()}
        self._default = default or []
        self._fail_times = fail_times
        self._error = error or SearchError("search provider down")
        self.calls: list[str] = []

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.calls.append(query)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._error
        return self._responses.get(query.strip().lower(), self._default)[:max_results]
