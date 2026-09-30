"""answer: stream the final markdown report, then parse it into a validated `Report`."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from research_agent.citations import replace_urls_with_citations
from research_agent.domain import Report, ResearchNote, Source
from research_agent.graph.context import GraphDeps, RunScope, StateNode, emitter
from research_agent.graph.nodes._common import record_error, track_node, truncate
from research_agent.graph.state import ResearchState
from research_agent.llm.base import Tier, Usage
from research_agent.prompts import load_prompt
from research_agent.report import Headings, build_report, headings_for, language_name

_NOTES_BUDGET_CHARS = 60_000
_SOURCE_EXCERPT_CHARS = 300


def usable_notes(state: ResearchState) -> list[ResearchNote]:
    """This run's successful notes first, then earlier runs' (follow-up memory)."""
    run_id = state.get("run_id")
    ok = [n for n in state.get("notes", []) if not n.failed and n.notes]
    return [n for n in ok if n.run_id == run_id] + [n for n in ok if n.run_id != run_id]


def failed_sub_questions(state: ResearchState) -> list[str]:
    run_notes = [n for n in state.get("notes", []) if n.run_id == state.get("run_id")]
    succeeded = {n.sub_question_id for n in run_notes if not n.failed}
    plan = state.get("plan")
    return [sq.question for sq in (plan.sub_questions if plan else []) if sq.id not in succeeded]


def _select(state: ResearchState) -> tuple[list[ResearchNote], list[Source]]:
    notes, used = [], 0
    for n in usable_notes(state):
        if used + len(n.notes) > _NOTES_BUDGET_CHARS:
            break
        notes.append(n)
        used += len(n.notes)
    urls = {u for n in notes for u in n.source_urls}
    sources = [s for s in state.get("sources", []) if s.url in urls]
    return notes, sources


def _notes_markdown(notes: list[ResearchNote], sources: list[Source]) -> str:
    return "\n\n".join(
        f"### {n.sub_question}\n{replace_urls_with_citations(n.notes, sources)}" for n in notes
    )


def make_answer(deps: GraphDeps) -> StateNode:
    prompt = load_prompt("answer")
    s = deps.settings

    async def answer(state: ResearchState, runtime: Runtime[RunScope]) -> dict[str, Any]:
        scope = deps.scope(runtime)
        emit = emitter(runtime)
        question = state["question"]
        plan = state.get("plan")
        language = plan.language if plan else "en"
        headings = headings_for(language)
        failed = failed_sub_questions(state)
        notes, sources = _select(state)
        async with track_node(scope, emit, "answer") as summary:
            if not notes:
                report = _degraded(question, failed, "every sub-question failed", language)
                summary.update(degraded=True)
                return {"report": report}

            user = prompt.render_user(
                question=question,
                language_name=language_name(language),
                summary_heading=headings.summary,
                key_findings_heading=headings.key_findings,
                open_questions_heading=headings.open_questions,
                failed=", ".join(failed) or "(none)",
                notes=_notes_markdown(notes, sources),
                sources="\n".join(
                    f"[{src.id}] {src.title} — {src.url}\n    "
                    f"{truncate(src.excerpt, _SOURCE_EXCERPT_CHARS)}"
                    for src in sources
                ),
            )
            chunks: list[str] = []
            try:
                scope.budget.ensure_tokens_available()
                async for item in deps.llm.stream_text(
                    system=prompt.system, user=user, tier=Tier.SMART, max_tokens=s.max_tokens_answer
                ):
                    if isinstance(item, Usage):
                        scope.record_llm("answer", item)
                    else:
                        chunks.append(item)
                        emit({"type": "token", "text": item})
                markdown = "".join(chunks)
            except Exception as exc:
                # Node boundary: fall back to the raw (already cited) notes.
                error = record_error(scope, "answer", exc)
                emit({"type": "answer_reset", "reason": "synthesis failed"})
                markdown = (
                    f"## {headings.summary}\n{headings.synthesis_failed.format(error=error)}"
                    f"\n\n{_notes_markdown(notes, sources)}"
                ).replace("\n### ", "\n## ")
            report = build_report(
                question=question,
                markdown=markdown,
                sources=sources,
                failed_sub_questions=failed,
                language=language,
            )
            summary.update(
                cited_sources=len(report.sources), invalid_citations=report.invalid_citations
            )
        return {"report": report}

    return answer


def _degraded(question: str, failed: list[str], reason: str, language: str) -> Report:
    h: Headings = headings_for(language)
    open_q = "\n".join(f"- {h.not_researched.format(question=q)}" for q in failed) or "- (none)"
    markdown = (
        f"## {h.summary}\n{h.research_failed.format(reason=reason)}\n\n"
        f"## {h.open_questions}\n{open_q}"
    )
    return build_report(
        question=question,
        markdown=markdown,
        sources=[],
        failed_sub_questions=failed,
        language=language,
    )
