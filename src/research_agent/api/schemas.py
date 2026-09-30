"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from research_agent.domain import Report, ResearchNote, Source

ThreadId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    search_provider: str


class ResearchRequest(BaseModel):
    question: Question
    thread_id: ThreadId | None = Field(
        default=None, description="Reuse to ask follow-ups on the same research thread."
    )


class ResearchResponse(BaseModel):
    thread_id: str
    run_id: str
    report: Report
    usage: dict[str, Any] = Field(description="Nodes, tool calls, tokens, cost, latency.")


class CheckpointOut(BaseModel):
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    created_at: str | None
    run_id: str | None


class ThreadResponse(BaseModel):
    thread_id: str
    questions: list[str]
    report: Report | None
    notes: list[ResearchNote]
    sources: list[Source]
    history: list[CheckpointOut]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
