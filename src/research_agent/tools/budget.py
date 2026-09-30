"""Per-run limits (searches, fetches, tokens) and a per-run search cache."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from research_agent.tools.errors import BudgetExceededError
from research_agent.tools.search import SearchProvider, SearchResult

__all__ = ["BudgetExceededError", "BudgetedSearch", "RunBudget"]


@dataclass
class RunBudget:
    """Mutable counters for one graph run.

    Parallel research branches share one instance. They all run on the same event loop and
    every check-and-increment below is synchronous, so no lock is needed.
    """

    max_searches: int
    max_fetches: int
    max_tokens: int
    searches_used: int = 0
    fetches_used: int = 0
    tokens_used: int = 0

    def consume_search(self) -> None:
        if self.searches_used >= self.max_searches:
            raise BudgetExceededError(f"search budget exhausted ({self.max_searches} per run)")
        self.searches_used += 1

    def consume_fetch(self) -> None:
        if self.fetches_used >= self.max_fetches:
            raise BudgetExceededError(f"fetch budget exhausted ({self.max_fetches} per run)")
        self.fetches_used += 1

    def add_tokens(self, n: int) -> None:
        self.tokens_used += n

    def ensure_tokens_available(self) -> None:
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceededError(f"token budget exhausted ({self.max_tokens} per run)")


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


@dataclass
class BudgetedSearch:
    """Wraps a SearchProvider with the run's search cap and a query cache."""

    provider: SearchProvider
    budget: RunBudget
    _cache: dict[tuple[str, int], list[SearchResult]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.provider.name

    def is_cached(self, query: str, max_results: int) -> bool:
        return (_normalize(query), max_results) in self._cache

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        key = (_normalize(query), max_results)
        if key in self._cache:
            return self._cache[key]
        self.budget.consume_search()
        results = await self.provider.search(query, max_results=max_results)
        self._cache[key] = results
        return results
