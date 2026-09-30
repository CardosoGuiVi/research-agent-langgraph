"""Conditional-edge functions. Pure: they only read state, so they are trivial to unit test."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from research_agent.graph.state import ResearchState, ResearchTask

MAX_QUERIES_PER_TASK = 3


def _normalize(q: str) -> str:
    return " ".join(q.lower().split())


def route_after_planner(state: ResearchState) -> list[Send] | Literal["answer"]:
    plan = state.get("plan")
    if plan is None or not plan.sub_questions:
        return "answer"  # follow-up fully covered by earlier findings (or nothing to research)
    return [
        Send(
            "research",
            ResearchTask(
                run_id=state["run_id"],
                question=state["question"],
                sub_question=sq,
                queries=sq.search_queries[:MAX_QUERIES_PER_TASK],
                iteration=state.get("iteration", 0) + 1,
            ),
        )
        for sq in plan.sub_questions
    ]


def route_after_analysis(
    state: ResearchState, *, max_iterations: int
) -> list[Send] | Literal["answer"]:
    """Back to research for sub-questions with gaps and new queries, else on to answer.

    The iteration cap is enforced here, in code, regardless of what the model suggests.
    """
    iteration = state.get("iteration", 0)
    plan = state.get("plan")
    if plan is None or iteration >= max_iterations:
        return "answer"
    by_id = {sq.id: sq for sq in plan.sub_questions}
    tried: dict[str, set[str]] = {}
    for note in state.get("notes", []):
        if note.run_id == state.get("run_id"):
            tried.setdefault(note.sub_question_id, set()).update(map(_normalize, note.queries))

    sends: list[Send] = []
    for a in state.get("assessments", []):
        sq = by_id.get(a.sub_question_id)
        if sq is None or a.coverage == "sufficient":
            continue
        seen = tried.get(sq.id, set())
        new_queries: list[str] = []
        for q in a.refined_queries:
            if _normalize(q) not in seen and q.strip():
                seen.add(_normalize(q))
                new_queries.append(q.strip())
        if not new_queries:
            continue
        sends.append(
            Send(
                "research",
                ResearchTask(
                    run_id=state["run_id"],
                    question=state["question"],
                    sub_question=sq,
                    queries=new_queries[:MAX_QUERIES_PER_TASK],
                    iteration=iteration + 1,
                ),
            )
        )
    return sends or "answer"
