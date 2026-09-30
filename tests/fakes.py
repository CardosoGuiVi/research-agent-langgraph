"""Test doubles shared across the suite."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from research_agent.config import Settings
from research_agent.container import Container, build_container
from research_agent.llm.base import LLMError, Tier, Usage
from research_agent.tools.errors import FetchError
from research_agent.tools.fetch import FetchedPage
from research_agent.tools.search import SearchError, SearchResult


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"anthropic_api_key": "sk-test", "tavily_api_key": None}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def build_test_container(
    *,
    llm: FakeLLM | None = None,
    search: FakeSearchProvider | None = None,
    fetcher: FakeFetcher | None = None,
    **settings_overrides: Any,
) -> Container:
    default_results = [
        SearchResult(title="OTel GenAI", url="https://otel.example/genai", snippet="semconv"),
        SearchResult(title="Langfuse", url="https://langfuse.example/docs", snippet="tracing"),
    ]
    return build_container(
        make_settings(**settings_overrides),
        llm=llm or FakeLLM(),
        search_provider=search or FakeSearchProvider(default=default_results),
        fetcher=fetcher or FakeFetcher(),
    )


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


# --- LLM ---------------------------------------------------------------------------------------
StructuredResponse = BaseModel | Exception | Callable[[str], BaseModel]
ToolPolicy = Callable[[Sequence[BaseMessage]], AIMessage]
_URL = re.compile(r"https?://[^\s)\]]+")


def default_research_policy(messages: Sequence[BaseMessage]) -> AIMessage:
    """search -> fetch first result -> write notes citing that URL."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    human = next(m for m in messages if isinstance(m, HumanMessage))
    if not tool_msgs:
        query = str(human.content).split("Suggested search queries:")[-1].strip().split(",")[0]
        call = {"name": "web_search", "args": {"query": query}, "id": "call_search"}
        return AIMessage(content="", tool_calls=[call])
    urls = _URL.findall(str(tool_msgs[0].content))
    if len(tool_msgs) == 1 and urls:
        call = {"name": "fetch_page", "args": {"url": urls[0]}, "id": "call_fetch"}
        return AIMessage(content="", tool_calls=[call])
    if not urls:
        return AIMessage(content="No usable results.\nGaps: everything")
    return AIMessage(content=f"- A key fact ({urls[0]})\nGaps: none")


class FakeLLM:
    """Scripted LLM port. Deterministic, no network, records every call."""

    def __init__(
        self,
        *,
        structured: dict[type[BaseModel], StructuredResponse | list[StructuredResponse]]
        | None = None,
        tool_policy: ToolPolicy = default_research_policy,
        answer: str | Exception = "## Summary\nAnswer [1].\n\n## Key Findings\n- Fact [1].",
        usage_tokens: tuple[int, int] = (100, 50),
    ) -> None:
        self._structured = {
            k: (v if isinstance(v, list) else [v]) for k, v in (structured or {}).items()
        }
        self._tool_policy = tool_policy
        self._answer = answer
        self._usage = usage_tokens
        self.calls: list[tuple[str, str]] = []  # (operation, tier)
        self.user_prompts: dict[str, list[str]] = {}

    def _u(self, tier: Tier) -> Usage:
        model = "claude-haiku-4-5" if tier is Tier.FAST else "claude-opus-5"
        return Usage(model=model, input_tokens=self._usage[0], output_tokens=self._usage[1])

    async def structured[T: BaseModel](
        self, schema: type[T], *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> tuple[T, Usage]:
        self.calls.append((f"structured:{schema.__name__}", tier.value))
        self.user_prompts.setdefault(schema.__name__, []).append(user)
        queue = self._structured.get(schema)
        if not queue:
            raise LLMError(f"FakeLLM has no scripted response for {schema.__name__}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        result = item(user) if callable(item) and not isinstance(item, BaseModel) else item
        return cast(T, result), self._u(tier)

    async def tool_step(
        self,
        messages: Sequence[BaseMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
        tier: Tier,
        max_tokens: int,
        allow_tool_calls: bool = True,
    ) -> tuple[AIMessage, Usage]:
        self.calls.append(("tool_step", tier.value))
        if not allow_tool_calls:  # forced wrap-up step
            return AIMessage(content="- Wrap-up notes\nGaps: step limit"), self._u(tier)
        return self._tool_policy(messages), self._u(tier)

    async def stream_text(
        self, *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> AsyncIterator[str | Usage]:
        self.calls.append(("stream_text", tier.value))
        self.user_prompts.setdefault("answer", []).append(user)
        if isinstance(self._answer, Exception):
            raise self._answer
        for i in range(0, len(self._answer), 16):
            yield self._answer[i : i + 16]
        yield self._u(tier)


class FakeFetcher:
    def __init__(self, pages: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.pages = pages or {}
        self.fail = fail
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        if self.fail or (self.pages and url not in self.pages):
            raise FetchError("fetch failed")
        return FetchedPage(
            url=url, title=f"Page {url}", text=self.pages.get(url, f"Content of {url}")
        )
