"""Typed graph state and reducers."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from research_agent.citations import SourceRegistry
from research_agent.domain import (
    Report,
    ResearchNote,
    ResearchPlan,
    Source,
    SubQuestion,
    SubQuestionAssessment,
)


def merge_sources(existing: list[Source] | None, new: list[Source] | None) -> list[Source]:
    """Reducer: dedupe by URL and assign citation ids in first-seen order.

    Research branches emit sources with placeholder ids; ids are assigned here, where
    LangGraph applies updates sequentially, so parallel branches never collide.
    """
    registry = SourceRegistry(existing or [])
    for src in new or []:
        registry.add(url=src.url, title=src.title, excerpt=src.excerpt, fetched=src.fetched)
    return registry.sources


class ResearchInput(TypedDict, total=False):
    """What callers pass to start (or continue) a run on a thread."""

    question: str
    run_id: str


class ResearchState(TypedDict, total=False):
    # Per run (reset by the planner at the start of every run)
    question: str
    run_id: str
    plan: ResearchPlan | None
    assessments: list[SubQuestionAssessment]
    iteration: int  # research rounds completed in this run
    report: Report | None
    # Accumulated across runs on the same thread (checkpointer memory)
    questions: Annotated[list[str], operator.add]
    notes: Annotated[list[ResearchNote], operator.add]
    sources: Annotated[list[Source], merge_sources]


class ResearchTask(TypedDict):
    """Payload sent (via `Send`) to one parallel research branch."""

    run_id: str
    question: str
    sub_question: SubQuestion
    queries: list[str]
    iteration: int
