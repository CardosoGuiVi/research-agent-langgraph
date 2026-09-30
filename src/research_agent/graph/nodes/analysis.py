"""analysis: judge coverage/conflicts per sub-question; routing then decides research vs answer."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from research_agent.domain import AnalysisOutput
from research_agent.graph.context import GraphDeps, RunScope, StateNode, emitter
from research_agent.graph.nodes._common import record_error, track_node, truncate
from research_agent.graph.state import ResearchState
from research_agent.llm.base import Tier
from research_agent.prompts import load_prompt

_NOTES_CHARS_PER_SUB_QUESTION = 3_000


def format_notes_for_analysis(state: ResearchState) -> str:
    plan = state.get("plan")
    run_notes = [n for n in state.get("notes", []) if n.run_id == state.get("run_id")]
    blocks: list[str] = []
    for sq in plan.sub_questions if plan else []:
        mine = [n for n in run_notes if n.sub_question_id == sq.id]
        tried = sorted({q for n in mine for q in n.queries})
        ok = [n.notes for n in mine if not n.failed and n.notes]
        failures = [n.error or "unknown error" for n in mine if n.failed]
        body = truncate("\n".join(ok), _NOTES_CHARS_PER_SUB_QUESTION) if ok else "(no notes)"
        status = f"\nFAILED: {'; '.join(failures)}" if failures and not ok else ""
        queries = ", ".join(tried) or "(none)"
        blocks.append(f"### {sq.id}: {sq.question}\nQueries tried: {queries}{status}\n{body}")
    return "\n\n".join(blocks)


def make_analysis(deps: GraphDeps) -> StateNode:
    prompt = load_prompt("analysis")
    s = deps.settings

    async def analysis(state: ResearchState, runtime: Runtime[RunScope]) -> dict[str, Any]:
        scope = deps.scope(runtime)
        iteration = state.get("iteration", 0) + 1
        async with track_node(scope, emitter(runtime), "analysis", iteration=iteration) as summary:
            if iteration >= s.max_iterations:
                # No research round can follow, so an assessment would not change anything.
                summary.update(skipped="iteration cap reached")
                return {"assessments": [], "iteration": iteration}
            try:
                scope.budget.ensure_tokens_available()
                output, usage = await deps.llm.structured(
                    AnalysisOutput,
                    system=prompt.system,
                    user=prompt.render_user(
                        question=state["question"], notes=format_notes_for_analysis(state)
                    ),
                    tier=Tier.FAST,
                    max_tokens=s.max_tokens_analysis,
                )
                scope.record_llm("analysis", usage)
            except Exception as exc:
                # Node boundary: without an assessment we still have notes; go write the report.
                record_error(scope, "analysis", exc)
                return {"assessments": [], "iteration": iteration}
            summary.update(coverage={a.sub_question_id: a.coverage for a in output.assessments})
        return {"assessments": output.assessments, "iteration": iteration}

    return analysis
