"""Application service: runs the graph for the API (invoke, stream, inspect threads)."""

from __future__ import annotations

import asyncio
import uuid
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from research_agent.domain import Report, ResearchNote, Source
from research_agent.graph.builder import ResearchGraph
from research_agent.graph.context import GraphDeps, RunScope
from research_agent.graph.state import ResearchInput
from research_agent.logging import get_logger
from research_agent.prompts import PROMPT_NAMES, load_prompt

_log = get_logger(__name__)


class ThreadBusyError(Exception):
    """Another run is in progress on this thread (runs on a thread are serialized)."""


@dataclass(frozen=True)
class RunResult:
    thread_id: str
    run_id: str
    report: Report
    usage: dict[str, Any]


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    created_at: str | None
    run_id: str | None


@dataclass(frozen=True)
class ThreadView:
    thread_id: str
    questions: list[str]
    report: Report | None
    notes: list[ResearchNote]
    sources: list[Source]
    history: list[Checkpoint]


def _prompt_versions() -> dict[str, int]:
    return {name: load_prompt(name).version for name in PROMPT_NAMES}


class ResearchService:
    def __init__(self, graph: ResearchGraph, deps: GraphDeps) -> None:
        self.graph = graph
        self.deps = deps
        # Weak values: a lock disappears once no run holds it, so this never grows unbounded.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def thread_lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    async def acquire(self, thread_id: str) -> asyncio.Lock:
        """Reserve a thread for one run; raises ThreadBusyError instead of queueing."""
        lock = self.thread_lock(thread_id)
        if lock.locked():
            raise ThreadBusyError(thread_id)
        await lock.acquire()
        return lock

    def _start(
        self, question: str, thread_id: str
    ) -> tuple[RunScope, RunnableConfig, ResearchInput]:
        run_id = uuid.uuid4().hex
        scope = self.deps.new_scope(run_id)
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "run_id": uuid.UUID(run_id),
        }
        structlog.contextvars.bind_contextvars(run_id=run_id, thread_id=thread_id)
        _log.info(
            "run_started",
            question_chars=len(question),
            prompts=_prompt_versions(),
            search_provider=self.deps.search_provider.name,
        )
        return scope, config, ResearchInput(question=question, run_id=run_id)

    def _finish(self, scope: RunScope, *, ok: bool) -> dict[str, Any]:
        usage = scope.metrics.summary()
        usage["searches"] = scope.budget.searches_used
        usage["fetches"] = scope.budget.fetches_used
        _log.info("run_finished", ok=ok, **usage)
        structlog.contextvars.unbind_contextvars("run_id", "thread_id")
        return usage

    async def run(self, question: str, thread_id: str) -> RunResult:
        lock = await self.acquire(thread_id)
        scope, config, graph_input = self._start(question, thread_id)
        ok = False
        try:
            state = await self.graph.ainvoke(graph_input, config, context=scope)
            report = state.get("report")
            if report is None:  # pragma: no cover - the answer node always sets a report
                raise RuntimeError("graph finished without a report")
            ok = True
            return RunResult(thread_id, scope.run_id, report, self._finish(scope, ok=True))
        finally:
            if not ok:
                self._finish(scope, ok=False)
            lock.release()

    async def stream(
        self, question: str, thread_id: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (event, data) pairs. The caller must hold the thread (see `acquire`)."""
        scope, config, graph_input = self._start(question, thread_id)
        ok = False
        try:
            yield "run_started", {"thread_id": thread_id, "run_id": scope.run_id}
            async for event in self.graph.astream(
                graph_input,
                config,
                context=scope,
                stream_mode="custom",
            ):
                data = dict(event)
                yield str(data.pop("type", "message")), data
            snapshot = await self.graph.aget_state(config)
            report: Report | None = snapshot.values.get("report")
            ok = report is not None
            usage = self._finish(scope, ok=ok)
            if report is None:  # pragma: no cover
                yield "error", {"message": "run finished without a report"}
                return
            yield (
                "report",
                {
                    "report": report.model_dump(mode="json"),
                    "usage": usage,
                    "run_id": scope.run_id,
                    "thread_id": thread_id,
                },
            )
            yield "done", {}
        except Exception:
            _log.exception("run_failed")
            if not ok:
                self._finish(scope, ok=False)
            yield "error", {"message": "internal error; see server logs"}

    async def get_thread(self, thread_id: str) -> ThreadView | None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values
        if not values:
            return None
        history: list[Checkpoint] = []
        async for snap in self.graph.aget_state_history(config, limit=50):
            meta = snap.metadata or {}
            history.append(
                Checkpoint(
                    checkpoint_id=str(snap.config["configurable"].get("checkpoint_id", "")),
                    step=int(meta.get("step", -1)),
                    source=str(meta.get("source", "")),
                    next=list(snap.next),
                    created_at=snap.created_at,
                    run_id=str(meta["run_id"]) if meta.get("run_id") else None,
                )
            )
        return ThreadView(
            thread_id=thread_id,
            questions=list(values.get("questions", [])),
            report=values.get("report"),
            notes=list(values.get("notes", [])),
            sources=list(values.get("sources", [])),
            history=history,
        )
