"""planner: break the question into 3-6 sub-questions with search queries (structured output)."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.runtime import Runtime

from research_agent.domain import PlannerOutput, ResearchPlan, SubQuestion
from research_agent.graph.context import GraphDeps, RunScope, StateNode, emitter
from research_agent.graph.nodes._common import prior_notes, record_error, track_node, truncate
from research_agent.graph.state import ResearchState
from research_agent.llm.base import Tier
from research_agent.prompts import load_prompt

_PRIOR_CONTEXT_CHARS = 6_000


def _prior_context(state: ResearchState, run_id: str) -> str:
    notes = prior_notes(state.get("notes", []), run_id)
    if not notes:
        return "(none)"
    lines = [f"- {n.sub_question}\n  {truncate(n.notes, 400)}" for n in notes]
    return truncate("\n".join(lines), _PRIOR_CONTEXT_CHARS)


def make_planner(deps: GraphDeps) -> StateNode:
    prompt = load_prompt("planner")
    settings = deps.settings

    async def planner(state: ResearchState, runtime: Runtime[RunScope]) -> dict[str, Any]:
        scope = deps.scope(runtime, reset=True)
        run_id = state.get("run_id") or scope.run_id or uuid.uuid4().hex
        question = state["question"].strip()
        has_prior = bool(prior_notes(state.get("notes", []), run_id))
        async with track_node(scope, emitter(runtime), "planner") as summary:
            try:
                scope.budget.ensure_tokens_available()
                output, usage = await deps.llm.structured(
                    PlannerOutput,
                    system=prompt.system,
                    user=prompt.render_user(
                        question=question,
                        prior_context=_prior_context(state, run_id),
                        min_n=settings.min_sub_questions,
                        max_n=settings.max_sub_questions,
                    ),
                    tier=Tier.FAST,
                    max_tokens=settings.max_tokens_planner,
                )
                scope.record_llm("planner", usage)
                planned, strategy = output.sub_questions, output.strategy
            except Exception as exc:
                # Node boundary: LLMError, BudgetExceededError or a provider error that survived
                # the SDK's retries. Degrade to researching the question directly.
                record_error(scope, "planner", exc)
                planned, strategy = [], "Planner unavailable; researching the question directly."

            sub_questions = [
                SubQuestion(
                    id=f"sq{i}",
                    question=p.question.strip(),
                    search_queries=[q.strip() for q in p.search_queries if q.strip()][:3]
                    or [p.question.strip()],
                    rationale=p.rationale,
                )
                for i, p in enumerate(planned[: settings.max_sub_questions], start=1)
                if p.question.strip()
            ]
            if not sub_questions and not has_prior:
                sub_questions = [
                    SubQuestion(id="sq1", question=question, search_queries=[question])
                ]
            plan = ResearchPlan(sub_questions=sub_questions, strategy=strategy)
            summary.update(sub_questions=[sq.question for sq in sub_questions])

        return {
            "run_id": run_id,
            "question": question,
            "questions": [question],
            "plan": plan,
            "assessments": [],
            "iteration": 0,
            "report": None,
        }

    return planner
