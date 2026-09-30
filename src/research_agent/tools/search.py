"""SearchProvider interface and its Tavily / DuckDuckGo implementations."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from research_agent.tools.errors import SearchError, TransientSearchError
from research_agent.tools.retry import call_with_retries

__all__ = [
    "DuckDuckGoSearch",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "TavilySearch",
]


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = Field(default="", description="Short excerpt returned by the provider.")


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


def _is_transient_status(status: int | None) -> bool:
    return status is not None and (status == 429 or status >= 500)


class TavilySearch:
    """Tavily search API (default when TAVILY_API_KEY is set)."""

    name = "tavily"

    def __init__(
        self, client: Any, *, timeout_s: float, attempts: int, base_delay_s: float = 0.5
    ) -> None:
        # `client` is a tavily.AsyncTavilyClient (typed as Any: the SDK ships no type hints).
        self._client = client
        self._timeout_s = timeout_s
        self._attempts = attempts
        self._base_delay_s = base_delay_s

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        async def once() -> list[SearchResult]:
            try:
                resp = await self._client.search(
                    query, max_results=max_results, search_depth="basic", timeout=self._timeout_s
                )
            except TimeoutError:
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                name = type(exc).__name__.lower()
                if _is_transient_status(status) or "timeout" in name or "connect" in name:
                    raise TransientSearchError(
                        f"tavily transient error: {type(exc).__name__}"
                    ) from exc
                raise SearchError(f"tavily error: {type(exc).__name__}") from exc
            return [
                SearchResult(
                    title=str(r.get("title") or r.get("url", "")),
                    url=str(r["url"]),
                    snippet=str(r.get("content") or ""),
                )
                for r in resp.get("results", [])
                if r.get("url")
            ]

        return await call_with_retries(
            once,
            attempts=self._attempts,
            timeout_s=self._timeout_s,
            base_delay_s=self._base_delay_s,
        )


class DuckDuckGoSearch:
    """DuckDuckGo via the `ddgs` library: no API key, used when Tavily is not configured."""

    name = "duckduckgo"

    def __init__(
        self, ddgs: Any, *, timeout_s: float, attempts: int, base_delay_s: float = 0.5
    ) -> None:
        # `ddgs` is a ddgs.DDGS instance (synchronous; run in a worker thread).
        self._ddgs = ddgs
        self._timeout_s = timeout_s
        self._attempts = attempts
        self._base_delay_s = base_delay_s

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        async def once() -> list[SearchResult]:
            try:
                rows = await asyncio.to_thread(self._ddgs.text, query, max_results=max_results)
            except Exception as exc:
                # ddgs raises RatelimitException / TimeoutException / DDGSException ("No results").
                name = type(exc).__name__.lower()
                if "ratelimit" in name or "timeout" in name:
                    raise TransientSearchError(
                        f"ddgs transient error: {type(exc).__name__}"
                    ) from exc
                if "no results" in str(exc).lower():
                    return []
                raise SearchError(f"ddgs error: {type(exc).__name__}") from exc
            return [
                SearchResult(
                    title=str(r.get("title") or r.get("href", "")),
                    url=str(r["href"]),
                    snippet=str(r.get("body") or ""),
                )
                for r in rows or []
                if r.get("href")
            ]

        return await call_with_retries(
            once,
            attempts=self._attempts,
            timeout_s=self._timeout_s,
            base_delay_s=self._base_delay_s,
        )
