from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from research_agent.domain import (
    AnalysisOutput,
    PlannedSubQuestion,
    PlannerOutput,
    SubQuestionAssessment,
)
from research_agent.graph.builder import build_graph, new_checkpointer
from research_agent.graph.context import GraphDeps
from research_agent.tools.search import SearchResult
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchProvider, make_settings

PLAN = PlannerOutput(
    sub_questions=[
        PlannedSubQuestion(question="Tracing tools?", search_queries=["llm tracing"], rationale=""),
        PlannedSubQuestion(question="Eval tools?", search_queries=["llm evals"], rationale=""),
        PlannedSubQuestion(question="Cost tracking?", search_queries=["llm cost"], rationale=""),
    ],
    strategy="s",
)
SEARCH = FakeSearchProvider(
    {
        "llm tracing": [SearchResult(title="T", url="https://t.example", snippet="t")],
        "llm evals": [SearchResult(title="E", url="https://e.example", snippet="e")],
        "llm cost": [SearchResult(title="C", url="https://c.example", snippet="c")],
        "llm cost dashboards": [SearchResult(title="C2", url="https://c2.example", snippet="c2")],
        "llm tracing follow-up": [SearchResult(title="F", url="https://f.example", snippet="f")],
    }
)


def _deps(llm: FakeLLM, search: FakeSearchProvider | None = None, **settings: Any) -> GraphDeps:
    return GraphDeps(
        settings=make_settings(**settings),
        llm=llm,
        search_provider=search or SEARCH,
        fetcher=FakeFetcher(),
    )


def _analysis(*items: tuple[str, str, list[str]]) -> AnalysisOutput:
    return AnalysisOutput(
        assessments=[
            SubQuestionAssessment(sub_question_id=i, coverage=c, refined_queries=q)  # type: ignore[arg-type]
            for i, c, q in items
        ]
    )


async def test_full_run_with_one_refinement_loop() -> None:
    llm = FakeLLM(
        structured={
            PlannerOutput: PLAN,
            AnalysisOutput: [
                _analysis(
                    ("sq1", "sufficient", []),
                    ("sq2", "sufficient", []),
                    ("sq3", "partial", ["llm cost dashboards"]),
                ),
                _analysis(("sq3", "sufficient", [])),
            ],
        },
        answer="## Summary\nTracing [1], evals [2], cost [3][4].\n",
    )
    deps = _deps(llm, max_iterations=3)
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    state = await graph.ainvoke(
        {"question": "Observability?", "run_id": "r1"}, config, context=deps.new_scope("r1")
    )

    report = state["report"]
    assert report is not None
    assert state["iteration"] == 2
    research_notes = [n for n in state["notes"] if n.run_id == "r1"]
    assert sorted((n.sub_question_id, n.iteration) for n in research_notes) == [
        ("sq1", 1),
        ("sq2", 1),
        ("sq3", 1),
        ("sq3", 2),
    ]
    # Citation ids are assigned deterministically at fan-in, in plan order.
    assert [(s.id, s.url) for s in state["sources"]] == [
        (1, "https://t.example"),
        (2, "https://e.example"),
        (3, "https://c.example"),
        (4, "https://c2.example"),
    ]
    assert [s.id for s in report.sources] == [1, 2, 3, 4]


async def test_iteration_cap_stops_the_loop_even_if_llm_wants_more() -> None:
    always_partial = _analysis(("sq3", "partial", ["llm cost dashboards"]))
    llm = FakeLLM(structured={PlannerOutput: PLAN, AnalysisOutput: always_partial})
    deps = _deps(llm, max_iterations=1)
    graph = build_graph(deps)
    state = await graph.ainvoke({"question": "Q", "run_id": "r1"}, context=deps.new_scope("r1"))
    assert state["iteration"] == 1
    assert ("structured:AnalysisOutput", "fast") not in llm.calls
    assert state["report"] is not None


async def test_follow_up_on_same_thread_reuses_prior_findings() -> None:
    follow_up_plan = PlannerOutput(
        sub_questions=[
            PlannedSubQuestion(
                question="More tracing?", search_queries=["llm tracing follow-up"], rationale=""
            )
        ],
        strategy="s",
    )
    llm = FakeLLM(
        structured={
            PlannerOutput: [PLAN, follow_up_plan],
            AnalysisOutput: _analysis(("sq1", "sufficient", [])),
        },
    )
    deps = _deps(llm, max_iterations=2)
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-follow"}}
    await graph.ainvoke({"question": "Q1", "run_id": "r1"}, config, context=deps.new_scope("r1"))
    state = await graph.ainvoke(
        {"question": "Q2 follow-up", "run_id": "r2"}, config, context=deps.new_scope("r2")
    )

    assert state["questions"] == ["Q1", "Q2 follow-up"]
    assert {n.run_id for n in state["notes"]} == {"r1", "r2"}
    assert "Tracing tools?" in llm.user_prompts["PlannerOutput"][1]  # prior findings in context
    answer_prompt = llm.user_prompts["answer"][1]
    assert "More tracing?" in answer_prompt
    assert "Tracing tools?" in answer_prompt
    # Earlier sources keep their numbers; new ones are appended.
    assert state["sources"][0].url == "https://t.example"
    assert state["sources"][0].id == 1
    assert state["sources"][-1].url == "https://f.example"
    assert state["plan"].sub_questions[0].question == "More tracing?"  # per-run fields reset


async def test_search_down_everywhere_still_returns_a_report() -> None:
    llm = FakeLLM(structured={PlannerOutput: PLAN, AnalysisOutput: _analysis()})
    deps = _deps(llm, search=FakeSearchProvider(fail_times=1000), max_iterations=2)
    state = await build_graph(deps).ainvoke(
        {"question": "Q", "run_id": "r1"}, context=deps.new_scope("r1")
    )
    report = state["report"]
    assert report is not None
    assert len(report.failed_sub_questions) == 3
    assert ("stream_text", "smart") not in llm.calls


async def test_stream_emits_custom_progress_events() -> None:
    llm = FakeLLM(structured={PlannerOutput: PLAN, AnalysisOutput: _analysis()})
    deps = _deps(llm, max_iterations=2)
    graph = build_graph(deps)
    events = [
        chunk
        async for mode, chunk in graph.astream(
            {"question": "Q", "run_id": "r1"}, stream_mode=["custom"], context=deps.new_scope("r1")
        )
    ]
    types = {e["type"] for e in events}
    assert {"node", "search", "fetch", "token"} <= types


async def test_graph_runs_without_context_using_fallback_scope() -> None:
    """LangGraph Studio invokes the graph without `context`."""
    llm = FakeLLM(structured={PlannerOutput: PLAN, AnalysisOutput: _analysis()})
    deps = _deps(llm, max_iterations=2)
    state = await build_graph(deps, checkpointer=new_checkpointer()).ainvoke(
        {"question": "Q"}, {"configurable": {"thread_id": "studio"}}
    )
    assert state["report"] is not None
    assert state["run_id"]
