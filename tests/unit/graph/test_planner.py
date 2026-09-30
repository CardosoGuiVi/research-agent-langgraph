from research_agent.domain import PlannedSubQuestion, PlannerOutput, ResearchNote
from research_agent.graph.nodes.planner import make_planner
from research_agent.llm.base import LLMError
from tests.fakes import FakeLLM
from tests.unit.graph.conftest import make_harness


def _planned(n: int) -> PlannerOutput:
    return PlannerOutput(
        sub_questions=[
            PlannedSubQuestion(
                question=f"Sub {i}?", search_queries=[f"q{i}a", f"q{i}b"], rationale="r"
            )
            for i in range(n)
        ],
        strategy="broad then deep",
        language="en",
    )


async def test_planner_builds_plan_and_resets_run_fields() -> None:
    h = make_harness(FakeLLM(structured={PlannerOutput: _planned(3)}))
    out = await make_planner(h.deps)({"question": "What is X?", "run_id": "r1"}, runtime=h.runtime)
    plan = out["plan"]
    assert plan is not None
    assert [sq.id for sq in plan.sub_questions] == ["sq1", "sq2", "sq3"]
    assert plan.sub_questions[0].search_queries == ["q0a", "q0b"]
    assert out["iteration"] == 0
    assert out["assessments"] == []
    assert out["report"] is None
    assert out["questions"] == ["What is X?"]
    assert h.llm.calls == [("structured:PlannerOutput", "fast")]
    assert any(e["type"] == "node" and e["node"] == "planner" for e in h.events)
    assert h.scope.metrics.llm_calls[0].node == "planner"


async def test_planner_clamps_to_max_sub_questions() -> None:
    h = make_harness(FakeLLM(structured={PlannerOutput: _planned(9)}), max_sub_questions=4)
    out = await make_planner(h.deps)({"question": "Q", "run_id": "r1"}, runtime=h.runtime)
    assert len(out["plan"].sub_questions) == 4


async def test_planner_without_queries_uses_the_sub_question_as_query() -> None:
    planned = PlannerOutput(
        sub_questions=[PlannedSubQuestion(question="Only this?", search_queries=[], rationale="")],
        strategy="s",
        language="en",
    )
    h = make_harness(FakeLLM(structured={PlannerOutput: planned}))
    out = await make_planner(h.deps)({"question": "Q", "run_id": "r1"}, runtime=h.runtime)
    assert out["plan"].sub_questions[0].search_queries == ["Only this?"]


async def test_empty_plan_without_prior_research_falls_back_to_the_question() -> None:
    h = make_harness(FakeLLM(structured={PlannerOutput: _planned(0)}))
    out = await make_planner(h.deps)({"question": "What is X?", "run_id": "r1"}, runtime=h.runtime)
    assert [sq.question for sq in out["plan"].sub_questions] == ["What is X?"]


async def test_empty_plan_on_follow_up_reuses_prior_findings() -> None:
    prior = ResearchNote(
        run_id="r0",
        sub_question_id="sq1",
        sub_question="Old?",
        iteration=1,
        queries=["x"],
        notes="- old fact (https://a.example)",
    )
    h = make_harness(FakeLLM(structured={PlannerOutput: _planned(0)}))
    state = {"question": "And Y?", "run_id": "r1", "notes": [prior]}
    out = await make_planner(h.deps)(state, runtime=h.runtime)  # type: ignore[arg-type]
    assert out["plan"].sub_questions == []
    prompt = h.llm.user_prompts["PlannerOutput"][0]
    assert "Old?" in prompt
    assert "old fact" in prompt


async def test_planner_failure_degrades_to_single_sub_question() -> None:
    h = make_harness(FakeLLM(structured={PlannerOutput: LLMError("boom")}))
    out = await make_planner(h.deps)({"question": "What is X?", "run_id": "r1"}, runtime=h.runtime)
    assert [sq.question for sq in out["plan"].sub_questions] == ["What is X?"]
    assert h.scope.metrics.errors


async def test_planner_records_the_question_language() -> None:
    planned = _planned(3).model_copy(update={"language": "pt"})
    h = make_harness(FakeLLM(structured={PlannerOutput: planned}))
    out = await make_planner(h.deps)({"question": "O que é X?", "run_id": "r1"}, runtime=h.runtime)
    assert out["plan"].language == "pt"


async def test_planner_failure_defaults_language_to_english() -> None:
    h = make_harness(FakeLLM(structured={PlannerOutput: LLMError("boom")}))
    out = await make_planner(h.deps)({"question": "O que é X?", "run_id": "r1"}, runtime=h.runtime)
    assert out["plan"].language == "en"
