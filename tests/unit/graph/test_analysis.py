from research_agent.domain import (
    AnalysisOutput,
    ResearchNote,
    ResearchPlan,
    SubQuestion,
    SubQuestionAssessment,
)
from research_agent.graph.nodes.analysis import make_analysis
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


def _state() -> ResearchState:
    return {
        "question": "Q",
        "run_id": "r1",
        "plan": PLAN,
        "iteration": 0,
        "notes": [
            ResearchNote(
                run_id="r0",
                sub_question_id="sq1",
                sub_question="Old?",
                iteration=1,
                queries=["old"],
                notes="OLD NOTES",
            ),
            ResearchNote(
                run_id="r1",
                sub_question_id="sq1",
                sub_question="Tracing?",
                iteration=1,
                queries=["t"],
                notes="- tracing fact (https://a.example)",
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


async def test_analysis_stores_assessments_and_increments_iteration() -> None:
    output = AnalysisOutput(
        assessments=[
            SubQuestionAssessment(sub_question_id="sq1", coverage="sufficient"),
            SubQuestionAssessment(
                sub_question_id="sq2", coverage="insufficient", refined_queries=["llm evals"]
            ),
        ]
    )
    h = make_harness(FakeLLM(structured={AnalysisOutput: output}))
    out = await make_analysis(h.deps)(_state(), runtime=h.runtime)
    assert out["iteration"] == 1
    assert [a.coverage for a in out["assessments"]] == ["sufficient", "insufficient"]
    prompt = h.llm.user_prompts["AnalysisOutput"][0]
    assert "tracing fact" in prompt
    assert "FAILED" in prompt
    assert "503" in prompt
    assert "OLD NOTES" not in prompt  # only this run's notes are analysed


async def test_analysis_failure_degrades_to_no_assessments() -> None:
    h = make_harness(FakeLLM(structured={AnalysisOutput: LLMError("boom")}))
    out = await make_analysis(h.deps)(_state(), runtime=h.runtime)
    assert out == {"assessments": [], "iteration": 1}
    assert h.scope.metrics.errors


async def test_analysis_skips_llm_when_iteration_cap_reached() -> None:
    h = make_harness(FakeLLM(), max_iterations=1)
    out = await make_analysis(h.deps)(_state(), runtime=h.runtime)
    assert out == {"assessments": [], "iteration": 1}
    assert h.llm.calls == []
