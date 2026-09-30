from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from research_agent.graph.context import GraphDeps, RunScope
from research_agent.tools.search import SearchResult
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchProvider, make_settings

RESULTS = [
    SearchResult(title="OTel GenAI", url="https://otel.example/genai", snippet="semconv"),
    SearchResult(title="Langfuse", url="https://langfuse.example/docs", snippet="tracing"),
]


@dataclass
class Harness:
    deps: GraphDeps
    scope: RunScope
    llm: FakeLLM
    search: FakeSearchProvider
    fetcher: FakeFetcher
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def runtime(self) -> Runtime[RunScope]:
        return Runtime(context=self.scope, stream_writer=self.events.append)

    @property
    def config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": "t1"}}


def make_harness(
    llm: FakeLLM | None = None,
    search: FakeSearchProvider | None = None,
    fetcher: FakeFetcher | None = None,
    **settings: Any,
) -> Harness:
    llm = llm or FakeLLM()
    search = search or FakeSearchProvider(default=RESULTS)
    fetcher = fetcher or FakeFetcher()
    deps = GraphDeps(
        settings=make_settings(**settings), llm=llm, search_provider=search, fetcher=fetcher
    )
    return Harness(deps=deps, scope=deps.new_scope("r1"), llm=llm, search=search, fetcher=fetcher)


@pytest.fixture
def harness() -> Harness:
    return make_harness()
