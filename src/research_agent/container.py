"""Composition root: wires settings and adapters into the objects the API uses."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from research_agent.config import SearchProviderName, Settings
from research_agent.graph.builder import build_graph, new_checkpointer
from research_agent.graph.context import GraphDeps
from research_agent.llm.anthropic import AnthropicLLM
from research_agent.llm.base import LLM
from research_agent.service import ResearchService
from research_agent.tools.fetch import HttpPageFetcher, PageFetcher
from research_agent.tools.search import DuckDuckGoSearch, SearchProvider, TavilySearch


@dataclass
class Container:
    settings: Settings
    service: ResearchService
    _http_clients: list[httpx.AsyncClient] = field(default_factory=list)

    async def aclose(self) -> None:
        for client in self._http_clients:
            await client.aclose()


def make_search_provider(settings: Settings) -> SearchProvider:
    if settings.search_provider is SearchProviderName.TAVILY and settings.tavily_api_key:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        return TavilySearch(
            client, timeout_s=settings.search_timeout_s, attempts=settings.tool_max_attempts
        )
    from ddgs import DDGS

    return DuckDuckGoSearch(
        DDGS(timeout=int(settings.search_timeout_s)),
        timeout_s=settings.search_timeout_s,
        attempts=settings.tool_max_attempts,
    )


def build_container(
    settings: Settings,
    *,
    llm: LLM | None = None,
    search_provider: SearchProvider | None = None,
    fetcher: PageFetcher | None = None,
) -> Container:
    """Real adapters by default; tests inject fakes for any of them."""
    clients: list[httpx.AsyncClient] = []
    if fetcher is None:
        http = httpx.AsyncClient(timeout=settings.fetch_timeout_s)
        clients.append(http)
        fetcher = HttpPageFetcher(
            http,
            max_bytes=settings.fetch_max_bytes,
            max_chars=settings.fetch_max_chars,
            timeout_s=settings.fetch_timeout_s,
            attempts=settings.tool_max_attempts,
        )
    deps = GraphDeps(
        settings=settings,
        llm=llm or AnthropicLLM(settings),
        search_provider=search_provider or make_search_provider(settings),
        fetcher=fetcher,
    )
    graph = build_graph(deps, checkpointer=new_checkpointer())
    return Container(settings=settings, service=ResearchService(graph, deps), _http_clients=clients)
