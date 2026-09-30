"""Helpers shared by nodes: timing/logging/metrics around each node execution."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from research_agent.domain import ResearchNote
from research_agent.graph.context import Emit, RunScope
from research_agent.logging import get_logger

_log = get_logger("research_agent.graph")


@asynccontextmanager
async def track_node(
    scope: RunScope, emit: Emit, node: str, **fields: Any
) -> AsyncIterator[dict[str, Any]]:
    """Log + emit start/finish of a node. Yields a dict the node can add summary fields to."""
    scope.metrics.nodes.append(node)
    emit({"type": "node", "node": node, "status": "started", **fields})
    start = time.perf_counter()
    summary: dict[str, Any] = {}
    try:
        yield summary
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        _log.info(
            "node_finished",
            run_id=scope.run_id,
            node=node,
            latency_ms=latency_ms,
            **fields,
            **summary,
        )
        emit(
            {
                "type": "node",
                "node": node,
                "status": "finished",
                "latency_ms": latency_ms,
                **fields,
                **summary,
            }
        )


def record_error(scope: RunScope, node: str, exc: BaseException) -> str:
    message = f"{node}: {type(exc).__name__}: {exc}"[:300]
    scope.metrics.errors.append(message)
    _log.warning("node_degraded", run_id=scope.run_id, node=node, error=message)
    return message


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def prior_notes(notes: list[ResearchNote], run_id: str) -> list[ResearchNote]:
    return [n for n in notes if n.run_id != run_id and not n.failed and n.notes]
