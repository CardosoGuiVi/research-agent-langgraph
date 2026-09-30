from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from research_agent.domain import SubQuestion
from research_agent.graph.nodes.research import make_research
from research_agent.graph.state import ResearchTask
from research_agent.tools.errors import TransientSearchError
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchProvider
from tests.unit.graph.conftest import make_harness


def _task(queries: list[str] | None = None) -> ResearchTask:
    sq = SubQuestion(
        id="sq1", question="Which tracing standards exist?", search_queries=["otel genai"]
    )
    return ResearchTask(
        run_id="r1", question="Q", sub_question=sq, queries=queries or ["otel genai"], iteration=1
    )


async def test_happy_path_searches_fetches_and_writes_notes() -> None:
    h = make_harness()
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    [note] = out["notes"]
    assert not note.failed
    assert note.notes.startswith("- A key fact (https://otel.example/genai)")
    assert note.tool_calls == 2
    assert note.source_urls == ["https://otel.example/genai", "https://langfuse.example/docs"]
    urls = {s.url: s for s in out["sources"]}
    assert urls["https://otel.example/genai"].fetched
    assert not urls["https://langfuse.example/docs"].fetched
    assert h.search.calls == ["otel genai"]
    assert h.fetcher.calls == ["https://otel.example/genai"]
    kinds = [(e["type"], e.get("status")) for e in h.events]
    assert ("search", None) in kinds
    assert ("fetch", None) in kinds
    assert h.scope.metrics.tool_calls == {"web_search": 1, "fetch_page": 1}


async def test_search_provider_down_marks_note_failed_without_raising() -> None:
    search = FakeSearchProvider(fail_times=99, error=TransientSearchError("503"))
    h = make_harness(search=search)
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    [note] = out["notes"]
    assert note.failed
    assert note.error is not None
    assert "search" in note.error.lower()
    assert out["sources"] == []


async def test_empty_results_mark_note_failed() -> None:
    h = make_harness(search=FakeSearchProvider(default=[]))
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    [note] = out["notes"]
    assert note.failed
    assert note.error == "no sources found"


async def test_fetch_failure_is_reported_to_the_model_and_research_continues() -> None:
    h = make_harness(fetcher=FakeFetcher(fail=True))
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    [note] = out["notes"]
    assert not note.failed  # search results are still usable sources
    assert "https://otel.example/genai" in note.source_urls


async def test_step_cap_forces_a_wrap_up_without_tools() -> None:
    counter = iter(range(100))

    def always_search(messages: Sequence[BaseMessage]) -> AIMessage:
        call = {"name": "web_search", "args": {"query": f"q{next(counter)}"}, "id": "c"}
        return AIMessage(content="", tool_calls=[call])

    h = make_harness(llm=FakeLLM(tool_policy=always_search), max_research_steps=3)
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    [note] = out["notes"]
    assert note.tool_calls == 3
    assert "Wrap-up notes" in note.notes
    assert h.llm.calls.count(("tool_step", "fast")) == 4


async def test_search_budget_exhaustion_is_returned_as_tool_error() -> None:
    counter = iter(range(100))

    def always_search(messages: Sequence[BaseMessage]) -> AIMessage:
        call = {"name": "web_search", "args": {"query": f"q{next(counter)}"}, "id": "c"}
        return AIMessage(content="", tool_calls=[call])

    h = make_harness(
        llm=FakeLLM(tool_policy=always_search), max_research_steps=4, max_searches_per_run=2
    )
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    assert len(h.search.calls) == 2
    assert out["notes"][0].tool_calls == 4


async def test_unknown_tool_and_bad_args_are_handled() -> None:
    steps = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "rm_rf", "args": {}, "id": "a"}]),
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "b"}]),
            AIMessage(content="- nothing (https://otel.example/genai)"),
        ]
    )
    h = make_harness(llm=FakeLLM(tool_policy=lambda _m: next(steps)))
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    assert out["notes"][0].tool_calls == 2
    assert h.search.calls == []


async def test_llm_failure_marks_note_failed() -> None:
    def boom(_m: Sequence[BaseMessage]) -> AIMessage:
        raise RuntimeError("anthropic 529 overloaded")

    h = make_harness(llm=FakeLLM(tool_policy=boom))
    out = await make_research(h.deps)(_task(), runtime=h.runtime)
    assert out["notes"][0].failed
    assert h.scope.metrics.errors


async def test_parallel_tool_calls_in_one_turn_are_all_executed() -> None:
    steps = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "a"}, "id": "1"},
                    {"name": "web_search", "args": {"query": "b"}, "id": "2"},
                ],
            ),
            AIMessage(content="- done (https://otel.example/genai)"),
        ]
    )
    h = make_harness(llm=FakeLLM(tool_policy=lambda _m: next(steps)))
    await make_research(h.deps)(_task(), runtime=h.runtime)
    assert sorted(h.search.calls) == ["a", "b"]
