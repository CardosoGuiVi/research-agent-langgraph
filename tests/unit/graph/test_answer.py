from research_agent.domain import ResearchNote, ResearchPlan, Source, SubQuestion
from research_agent.graph.nodes.answer import make_answer
from research_agent.graph.state import ResearchState
from research_agent.llm.base import LLMError
from tests.fakes import FakeLLM
from tests.unit.graph.conftest import make_harness

PLAN = ResearchPlan(
    sub_questions=[
        SubQuestion(id="sq1", question="Tracing?", search_queries=["t"]),
        SubQuestion(id="sq2", question="Evals?", search_queries=["e"]),
    ],
    strategy="s",
)
SOURCES = [
    Source(id=1, url="https://a.example", title="A", excerpt="aaa", fetched=True),
    Source(id=2, url="https://b.example", title="B", excerpt="bbb"),
    Source(id=3, url="https://old.example", title="Old", excerpt="old"),
]
ANSWER = """## Summary
Tracing matters [1] and old context helps [3].

## Key Findings
- Tracing [1].
- Made up [7].

## Tracing
Details [1].

## Open Questions
- Evals were not researched.
"""


def _state(*, all_failed: bool = False) -> ResearchState:
    return {
        "question": "What about observability?",
        "run_id": "r1",
        "plan": PLAN,
        "iteration": 1,
        "sources": SOURCES,
        "notes": [
            ResearchNote(
                run_id="r0",
                sub_question_id="sq1",
                sub_question="Earlier?",
                iteration=1,
                queries=["x"],
                notes="- earlier (https://old.example)",
                source_urls=["https://old.example"],
            ),
            ResearchNote(
                run_id="r1",
                sub_question_id="sq1",
                sub_question="Tracing?",
                iteration=1,
                queries=["t"],
                notes="- tracing (https://a.example)",
                source_urls=["https://a.example", "https://b.example"],
                failed=all_failed,
                error="x" if all_failed else None,
            ),
            ResearchNote(
                run_id="r1",
                sub_question_id="sq2",
                sub_question="Evals?",
                iteration=1,
                queries=["e"],
                failed=True,
                error="search failed: 503",
            ),
        ],
    }


async def test_answer_streams_tokens_and_builds_report_with_citations() -> None:
    h = make_harness(FakeLLM(answer=ANSWER))
    out = await make_answer(h.deps)(_state(), runtime=h.runtime)
    report = out["report"]
    assert report.summary.startswith("Tracing matters [1]")
    assert [s.id for s in report.sources] == [1, 3]
    assert report.invalid_citations == [7]
    assert report.failed_sub_questions == ["Evals?"]
    tokens = "".join(e["text"] for e in h.events if e["type"] == "token")
    assert tokens == ANSWER
    assert h.llm.calls == [("stream_text", "smart")]
    prompt = h.llm.user_prompts["answer"][0]
    assert "- tracing [1]" in prompt  # URLs replaced by citation numbers
    assert "- earlier [3]" in prompt  # prior findings on the thread are reused
    assert "[1] A — https://a.example" in prompt
    assert "Evals?" in prompt  # failed sub-question is surfaced


async def test_all_research_failed_returns_degraded_report_without_llm() -> None:
    h = make_harness(FakeLLM())
    state = _state(all_failed=True)
    state["notes"] = [n for n in state["notes"] if n.run_id == "r1"]
    out = await make_answer(h.deps)(state, runtime=h.runtime)
    report = out["report"]
    assert h.llm.calls == []
    assert "could not be completed" in report.summary.lower()
    assert set(report.failed_sub_questions) == {"Tracing?", "Evals?"}
    assert report.sources == []


async def test_llm_failure_falls_back_to_raw_notes() -> None:
    h = make_harness(FakeLLM(answer=LLMError("overloaded")))
    out = await make_answer(h.deps)(_state(), runtime=h.runtime)
    report = out["report"]
    assert "synthesis step failed" in report.summary.lower()
    assert any("tracing [1]" in s.content for s in report.sections)
    assert [s.id for s in report.sources] == [1, 3]
    assert h.scope.metrics.errors
