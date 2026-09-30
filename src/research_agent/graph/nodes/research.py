"""research: a bounded tool-calling loop (web_search / fetch_page) for ONE sub-question.

Runs once per sub-question in parallel (LangGraph `Send`). A branch never raises: failures are
recorded on its `ResearchNote` so the run degrades gracefully instead of aborting.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime

from research_agent.domain import ResearchNote, Source
from research_agent.graph.context import Emit, GraphDeps, RunScope, emitter
from research_agent.graph.nodes._common import record_error, track_node, truncate
from research_agent.graph.state import ResearchTask
from research_agent.llm.anthropic import extract_text
from research_agent.llm.base import Tier
from research_agent.prompts import load_prompt
from research_agent.tools.errors import BudgetExceededError, SearchError, ToolError


class ResearchNode(Protocol):
    async def __call__(
        self, state: ResearchTask, *, runtime: Runtime[RunScope]
    ) -> dict[str, Any]: ...


WEB_SEARCH_TOOL: dict[str, Any] = {
    "name": "web_search",
    "description": "Search the web. Returns up to N results with title, URL and snippet.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Concise keyword query."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
FETCH_PAGE_TOOL: dict[str, Any] = {
    "name": "fetch_page",
    "description": "Fetch a web page and return its main readable text (truncated).",
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "An http(s) URL from results."}},
        "required": ["url"],
        "additionalProperties": False,
    },
}
TOOLS = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]
_WRAP_UP = (
    "You have used the tool budget for this sub-question. Write your research notes now, "
    "in the required format, using only what you have found."
)
_EXCERPT_CHARS = 600


class _BranchState:
    """Mutable bookkeeping for one branch: sources seen and tool outcomes."""

    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self.search_errors: list[str] = []
        self.tool_calls = 0

    def add_source(self, url: str, title: str, excerpt: str, *, fetched: bool) -> None:
        existing = self.sources.get(url)
        if existing is None or (fetched and not existing.fetched):
            # id=0 is a placeholder; the `merge_sources` reducer assigns citation ids.
            self.sources[url] = Source(
                id=0,
                url=url,
                title=title,
                excerpt=truncate(excerpt, _EXCERPT_CHARS),
                fetched=fetched,
            )


def make_research(deps: GraphDeps) -> ResearchNode:
    prompt = load_prompt("research")
    s = deps.settings

    async def run_tool(
        call: ToolCall, scope: RunScope, branch: _BranchState, emit: Emit, sq_id: str
    ) -> ToolMessage:
        name, args, call_id = call["name"], call.get("args") or {}, call.get("id") or ""
        branch.tool_calls += 1
        scope.metrics.tool_calls[name] += 1
        try:
            if name == "web_search":
                query = str(args["query"])
                emit(
                    {
                        "type": "search",
                        "sub_question_id": sq_id,
                        "query": query,
                        "cached": scope.search.is_cached(query, s.search_results_per_query),
                    }
                )
                results = await scope.search.search(query, max_results=s.search_results_per_query)
                for r in results:
                    branch.add_source(r.url, r.title, r.snippet, fetched=False)
                body = (
                    "\n\n".join(
                        f"[{i}] {r.title}\nURL: {r.url}\n{r.snippet}"
                        for i, r in enumerate(results, 1)
                    )
                    or "No results."
                )
            elif name == "fetch_page":
                url = str(args["url"])
                emit({"type": "fetch", "sub_question_id": sq_id, "url": url})
                scope.budget.consume_fetch()
                page = await deps.fetcher.fetch(url)
                branch.add_source(page.url, page.title, page.text, fetched=True)
                body = f"Title: {page.title}\nURL: {page.url}\n\n{page.text}"
            else:
                return ToolMessage(f"Unknown tool: {name}", tool_call_id=call_id, status="error")
        except KeyError as exc:
            return ToolMessage(f"Missing argument: {exc}", tool_call_id=call_id, status="error")
        except ToolError as exc:
            if isinstance(exc, SearchError):
                branch.search_errors.append(str(exc))
            kind = "budget" if isinstance(exc, BudgetExceededError) else "error"
            emit({"type": "tool_error", "sub_question_id": sq_id, "tool": name, "kind": kind})
            return ToolMessage(f"{name} failed: {exc}", tool_call_id=call_id, status="error")
        return ToolMessage(body, tool_call_id=call_id)

    async def research(state: ResearchTask, runtime: Runtime[RunScope]) -> dict[str, Any]:
        task = state  # the `Send` payload for this branch, not the shared graph state
        scope = deps.scope(runtime)
        emit = emitter(runtime)
        sq = task["sub_question"]
        branch = _BranchState()
        notes, error = "", None
        async with track_node(
            scope, emit, "research", sub_question_id=sq.id, iteration=task["iteration"]
        ) as summary:
            messages: list[BaseMessage] = [
                HumanMessage(
                    prompt.render_user(
                        question=task["question"],
                        sub_question=sq.question,
                        queries=", ".join(task["queries"]),
                    )
                )
            ]
            try:
                for _ in range(s.max_research_steps):
                    scope.budget.ensure_tokens_available()
                    ai, usage = await deps.llm.tool_step(
                        messages,
                        system=prompt.system,
                        tools=TOOLS,
                        tier=Tier.FAST,
                        max_tokens=s.max_tokens_research,
                    )
                    scope.record_llm("research", usage)
                    messages.append(ai)
                    if not ai.tool_calls:
                        notes = extract_text(ai)
                        break
                    # Independent tool calls from one turn run concurrently; all results go
                    # back together in the next request.
                    messages.extend(
                        await asyncio.gather(
                            *(run_tool(c, scope, branch, emit, sq.id) for c in ai.tool_calls)
                        )
                    )
                else:
                    ai, usage = await deps.llm.tool_step(
                        [*messages, HumanMessage(_WRAP_UP)],
                        system=prompt.system,
                        tools=TOOLS,
                        tier=Tier.FAST,
                        max_tokens=s.max_tokens_research,
                        allow_tool_calls=False,
                    )
                    scope.record_llm("research", usage)
                    notes = extract_text(ai)
            except Exception as exc:
                # Branch boundary: one failing sub-question must not abort the whole run.
                error = record_error(scope, f"research[{sq.id}]", exc)

            if error is None and not branch.sources:
                error = (
                    f"search failed: {branch.search_errors[-1]}"
                    if branch.search_errors
                    else "no sources found"
                )
            summary.update(
                tool_calls=branch.tool_calls, sources=len(branch.sources), failed=error is not None
            )

        note = ResearchNote(
            run_id=task["run_id"],
            sub_question_id=sq.id,
            sub_question=sq.question,
            iteration=task["iteration"],
            queries=task["queries"],
            notes=notes.strip(),
            source_urls=list(branch.sources),
            tool_calls=branch.tool_calls,
            failed=error is not None,
            error=error,
        )
        return {"notes": [note], "sources": list(branch.sources.values())}

    return research
