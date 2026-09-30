import pytest

from research_agent.tools.budget import BudgetedSearch, BudgetExceededError, RunBudget
from research_agent.tools.search import SearchError, SearchResult
from tests.fakes import FakeSearchProvider


def _results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(title=f"t{i}", url=f"https://example.com/{i}", snippet="s") for i in range(n)
    ]


async def test_repeated_query_is_served_from_cache_and_not_counted() -> None:
    provider = FakeSearchProvider({"llm observability": _results()})
    budget = RunBudget(max_searches=5, max_fetches=5, max_tokens=1000)
    search = BudgetedSearch(provider, budget)

    first = await search.search("LLM observability", max_results=5)
    second = await search.search("  llm   OBSERVABILITY ", max_results=5)

    assert first == second
    assert provider.calls == ["LLM observability"]
    assert budget.searches_used == 1


async def test_search_cap_is_enforced() -> None:
    provider = FakeSearchProvider(default=_results())
    budget = RunBudget(max_searches=2, max_fetches=5, max_tokens=1000)
    search = BudgetedSearch(provider, budget)

    await search.search("a", max_results=3)
    await search.search("b", max_results=3)
    with pytest.raises(BudgetExceededError):
        await search.search("c", max_results=3)
    assert provider.calls == ["a", "b"]


async def test_cached_query_still_allowed_after_cap() -> None:
    provider = FakeSearchProvider(default=_results())
    budget = RunBudget(max_searches=1, max_fetches=5, max_tokens=1000)
    search = BudgetedSearch(provider, budget)
    await search.search("a", max_results=3)
    assert await search.search("a", max_results=3)


async def test_failed_search_is_not_cached() -> None:
    provider = FakeSearchProvider(default=_results(), fail_times=1)
    budget = RunBudget(max_searches=5, max_fetches=5, max_tokens=1000)
    search = BudgetedSearch(provider, budget)
    with pytest.raises(SearchError):
        await search.search("a", max_results=3)
    assert await search.search("a", max_results=3)
    assert budget.searches_used == 2


def test_token_budget_tracking() -> None:
    budget = RunBudget(max_searches=1, max_fetches=1, max_tokens=100)
    budget.add_tokens(60)
    budget.ensure_tokens_available()
    budget.add_tokens(50)
    with pytest.raises(BudgetExceededError):
        budget.ensure_tokens_available()


def test_fetch_budget() -> None:
    budget = RunBudget(max_searches=1, max_fetches=1, max_tokens=100)
    budget.consume_fetch()
    with pytest.raises(BudgetExceededError):
        budget.consume_fetch()
