"""Domain models shared by the graph, the API and the eval harness."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Coverage = Literal["sufficient", "partial", "insufficient"]


# --- Planner ---------------------------------------------------------------------------------
class PlannedSubQuestion(BaseModel):
    """LLM output for one sub-question (ids are assigned by code, not the model)."""

    question: str = Field(description="A focused, self-contained sub-question.")
    search_queries: list[str] = Field(description="1-3 concise web search queries.")
    rationale: str = Field(description="Why this sub-question matters for the main question.")


class PlannerOutput(BaseModel):
    sub_questions: list[PlannedSubQuestion]
    strategy: str = Field(description="One or two sentences on the overall research strategy.")
    language: str = Field(
        description="ISO 639-1 code of the language the question is written in, e.g. 'en', 'pt'."
    )


class SubQuestion(BaseModel):
    id: str
    question: str
    search_queries: list[str]
    rationale: str = ""


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion]
    strategy: str
    language: str = "en"  # ISO 639-1 code of the question; the report is written in it


# --- Research --------------------------------------------------------------------------------
class Source(BaseModel):
    """A web source seen during research. `id` is the citation number used in reports."""

    id: int
    url: str
    title: str
    excerpt: str = Field(default="", description="Snippet or start of the fetched page text.")
    fetched: bool = Field(default=False, description="True if the page itself was read.")


class ResearchNote(BaseModel):
    """Output of one research branch (one sub-question, one iteration)."""

    run_id: str
    sub_question_id: str
    sub_question: str
    iteration: int
    queries: list[str]
    notes: str = ""
    source_urls: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    failed: bool = False
    error: str | None = None


# --- Analysis --------------------------------------------------------------------------------
class SubQuestionAssessment(BaseModel):
    sub_question_id: str
    coverage: Coverage
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    assessments: list[SubQuestionAssessment]


# --- Report ----------------------------------------------------------------------------------
class ReportSection(BaseModel):
    title: str
    content: str


class SourceRef(BaseModel):
    id: int
    url: str
    title: str
    fetched: bool = False


class Report(BaseModel):
    question: str
    language: str = Field(default="en", description="ISO 639-1 code the report is written in.")
    summary: str
    sections: list[ReportSection]
    key_findings: list[str]
    open_questions: list[str]
    sources: list[SourceRef] = Field(description="Only sources that are cited in the report.")
    failed_sub_questions: list[str] = Field(default_factory=list)
    invalid_citations: list[int] = Field(
        default_factory=list,
        description="Citation numbers the model produced that map to no source.",
    )
    markdown: str = Field(description="Full report as markdown, including the sources list.")
