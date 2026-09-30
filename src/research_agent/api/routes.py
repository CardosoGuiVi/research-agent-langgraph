"""HTTP routes."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from research_agent import __version__
from research_agent.api.schemas import (
    CheckpointOut,
    ErrorResponse,
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    ThreadId,
    ThreadResponse,
)
from research_agent.container import Container
from research_agent.service import ResearchService, ThreadBusyError

router = APIRouter()

_BUSY: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorResponse, "description": "A run is already active on this thread."}
}


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_service(container: Annotated[Container, Depends(get_container)]) -> ResearchService:
    return container.service


ServiceDep = Annotated[ResearchService, Depends(get_service)]


def _busy(thread_id: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, f"thread {thread_id} has a run in progress")


@router.get("/health", response_model=HealthResponse)
async def health(container: Annotated[Container, Depends(get_container)]) -> HealthResponse:
    return HealthResponse(
        status="ok", version=__version__, search_provider=container.settings.search_provider.value
    )


@router.post("/research", response_model=ResearchResponse, responses=_BUSY)
async def research(body: ResearchRequest, service: ServiceDep) -> ResearchResponse:
    """Run the full research graph and return the final report."""
    thread_id = body.thread_id or uuid.uuid4().hex
    try:
        result = await service.run(body.question, thread_id)
    except ThreadBusyError:
        raise _busy(thread_id) from None
    return ResearchResponse(
        thread_id=result.thread_id, run_id=result.run_id, report=result.report, usage=result.usage
    )


async def reserve_thread(body: ResearchRequest, service: ServiceDep) -> AsyncIterator[str]:
    """Hold the thread for the whole streamed response (409 before any byte is sent)."""
    thread_id = body.thread_id or uuid.uuid4().hex
    try:
        lock = await service.acquire(thread_id)
    except ThreadBusyError:
        raise _busy(thread_id) from None
    try:
        yield thread_id
    finally:
        lock.release()


@router.post("/research/stream", response_class=EventSourceResponse, responses=_BUSY)
async def research_stream(
    body: ResearchRequest,
    service: ServiceDep,
    thread_id: Annotated[str, Depends(reserve_thread)],
) -> AsyncIterator[ServerSentEvent]:
    """Server-Sent Events: run_started, node, search, fetch, tool_error, token, report, done.

    `error` is sent instead of `report` if the run fails unexpectedly.
    """
    async for name, data in service.stream(body.question, thread_id):
        yield ServerSentEvent(event=name, data=data)


@router.get(
    "/threads/{thread_id}", response_model=ThreadResponse, responses={404: {"model": ErrorResponse}}
)
async def get_thread(thread_id: Annotated[ThreadId, Path()], service: ServiceDep) -> ThreadResponse:
    """Accumulated state of a research thread plus its checkpoint history."""
    view = await service.get_thread(thread_id)
    if view is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "thread not found")
    return ThreadResponse(
        thread_id=view.thread_id,
        questions=view.questions,
        report=view.report,
        notes=view.notes,
        sources=view.sources,
        history=[CheckpointOut(**asdict(c)) for c in view.history],
    )
