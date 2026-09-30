from langgraph.types import Send

from research_agent.domain import ResearchNote, ResearchPlan, SubQuestion, SubQuestionAssessment
from research_agent.graph.routing import route_after_analysis, route_after_planner
from research_agent.graph.state import ResearchState


def _plan(n: int = 2) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=[
            SubQuestion(id=f"sq{i}", question=f"Q{i}?", search_queries=[f"q{i}"])
            for i in range(1, n + 1)
        ],
        strategy="s",
    )


def _note(sq: str, queries: list[str], iteration: int = 1) -> ResearchNote:
    return ResearchNote(
        run_id="r1", sub_question_id=sq, sub_question="?", iteration=iteration, queries=queries
    )


def _assess(sq: str, coverage: str, refined: list[str] | None = None) -> SubQuestionAssessment:
    return SubQuestionAssessment(
        sub_question_id=sq,
        coverage=coverage,
        refined_queries=refined or [],  # type: ignore[arg-type]
    )


def test_planner_fans_out_one_send_per_sub_question() -> None:
    state: ResearchState = {"question": "Q", "run_id": "r1", "plan": _plan(3), "iteration": 0}
    sends = route_after_planner(state)
    assert isinstance(sends, list)
    assert all(isinstance(s, Send) and s.node == "research" for s in sends)
    assert [s.arg["sub_question"].id for s in sends] == ["sq1", "sq2", "sq3"]
    assert sends[0].arg["queries"] == ["q1"]
    assert sends[0].arg["iteration"] == 1


def test_planner_with_empty_plan_goes_to_answer() -> None:
    state: ResearchState = {"question": "Q", "run_id": "r1", "plan": _plan(0), "iteration": 0}
    assert route_after_planner(state) == "answer"


def test_analysis_all_sufficient_goes_to_answer() -> None:
    state: ResearchState = {
        "question": "Q",
        "run_id": "r1",
        "plan": _plan(2),
        "iteration": 1,
        "assessments": [_assess("sq1", "sufficient"), _assess("sq2", "sufficient")],
    }
    assert route_after_analysis(state, max_iterations=3) == "answer"


def test_analysis_sends_only_gapped_sub_questions_with_refined_queries() -> None:
    state: ResearchState = {
        "question": "Q",
        "run_id": "r1",
        "plan": _plan(3),
        "iteration": 1,
        "notes": [_note("sq2", ["q2"])],
        "assessments": [
            _assess("sq1", "sufficient"),
            _assess("sq2", "partial", ["q2", "new q2"]),  # "q2" already tried -> dropped
            _assess("sq3", "insufficient", []),  # nothing new to try -> not retried
        ],
    }
    sends = route_after_analysis(state, max_iterations=3)
    assert isinstance(sends, list)
    assert [(s.arg["sub_question"].id, s.arg["queries"], s.arg["iteration"]) for s in sends] == [
        ("sq2", ["new q2"], 2)
    ]


def test_iteration_cap_forces_answer_even_with_gaps() -> None:
    state: ResearchState = {
        "question": "Q",
        "run_id": "r1",
        "plan": _plan(1),
        "iteration": 2,
        "assessments": [_assess("sq1", "insufficient", ["another query"])],
    }
    assert route_after_analysis(state, max_iterations=2) == "answer"


def test_unknown_sub_question_ids_from_llm_are_ignored() -> None:
    state: ResearchState = {
        "question": "Q",
        "run_id": "r1",
        "plan": _plan(1),
        "iteration": 1,
        "assessments": [_assess("sq99", "insufficient", ["x"])],
    }
    assert route_after_analysis(state, max_iterations=3) == "answer"


def test_refined_queries_are_capped_at_three() -> None:
    state: ResearchState = {
        "question": "Q",
        "run_id": "r1",
        "plan": _plan(1),
        "iteration": 1,
        "assessments": [_assess("sq1", "partial", ["a", "b", "c", "d"])],
    }
    sends = route_after_analysis(state, max_iterations=3)
    assert isinstance(sends, list)
    assert sends[0].arg["queries"] == ["a", "b", "c"]
