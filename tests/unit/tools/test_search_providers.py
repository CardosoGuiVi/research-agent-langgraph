import time
from typing import Any

import pytest

from research_agent.tools.errors import SearchError, TransientSearchError
from research_agent.tools.search import DuckDuckGoSearch, SearchProvider, TavilySearch


class StubTavily:
    def __init__(self, *, fail: list[Exception] | None = None) -> None:
        self.fail = list(fail or [])
        self.calls = 0

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        return {
            "results": [
                {"title": "A", "url": "https://a.example", "content": "alpha"},
                {"title": None, "url": "https://b.example", "content": None},
                {"title": "no url"},
            ]
        }


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class HTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = _Resp(status)


class StubDDGS:
    def __init__(self, *, rows: list[dict[str, str]] | None = None, exc: Exception | None = None):
        self.rows = rows or []
        self.exc = exc
        self.calls = 0

    def text(self, query: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.rows


class RatelimitException(Exception):  # noqa: N818 - mirrors the ddgs class name
    pass


class DDGSException(Exception):  # noqa: N818 - mirrors the ddgs class name
    pass


async def test_tavily_maps_results_and_skips_rows_without_url() -> None:
    provider = TavilySearch(StubTavily(), timeout_s=1, attempts=1)
    assert isinstance(provider, SearchProvider)
    results = await provider.search("q", max_results=5)
    assert [r.url for r in results] == ["https://a.example", "https://b.example"]
    assert results[0].snippet == "alpha"
    assert results[1].title == "https://b.example"


async def test_tavily_retries_5xx_then_succeeds() -> None:
    stub = StubTavily(fail=[HTTPError(503)])
    provider = TavilySearch(stub, timeout_s=1, attempts=2, base_delay_s=0)
    provider_results = await provider.search("q", max_results=5)
    assert provider_results
    assert stub.calls == 2


async def test_tavily_4xx_is_not_retried() -> None:
    stub = StubTavily(fail=[HTTPError(401), HTTPError(401)])
    provider = TavilySearch(stub, timeout_s=1, attempts=3)
    with pytest.raises(SearchError) as info:
        await provider.search("q", max_results=5)
    assert not isinstance(info.value, TransientSearchError)
    assert stub.calls == 1


async def test_ddg_maps_results() -> None:
    stub = StubDDGS(rows=[{"title": "T", "href": "https://x.example", "body": "b"}, {"title": "x"}])
    results = await DuckDuckGoSearch(stub, timeout_s=1, attempts=1).search("q", max_results=3)
    assert [(r.title, r.url, r.snippet) for r in results] == [("T", "https://x.example", "b")]


async def test_ddg_rate_limit_is_transient_and_retried() -> None:
    stub = StubDDGS(exc=RatelimitException("slow down"))
    with pytest.raises(TransientSearchError):
        await DuckDuckGoSearch(stub, timeout_s=1, attempts=2, base_delay_s=0).search(
            "q", max_results=3
        )
    assert stub.calls == 2


async def test_ddg_no_results_is_empty_list() -> None:
    stub = StubDDGS(exc=DDGSException("No results found."))
    assert await DuckDuckGoSearch(stub, timeout_s=1, attempts=1).search("q", max_results=3) == []


async def test_ddg_other_errors_are_search_errors() -> None:
    stub = StubDDGS(exc=DDGSException("boom"))
    with pytest.raises(SearchError):
        await DuckDuckGoSearch(stub, timeout_s=1, attempts=1).search("q", max_results=3)


class SlowDDGS:
    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        time.sleep(0.2)
        return []


async def test_ddg_hanging_search_is_a_search_error() -> None:
    """Seen in a live eval: ddgs hung under rate limiting and a raw TimeoutError killed a branch."""
    with pytest.raises(TransientSearchError, match="timed out"):
        await DuckDuckGoSearch(SlowDDGS(), timeout_s=0.01, attempts=1).search("q", max_results=3)
