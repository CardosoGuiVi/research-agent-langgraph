"""Deterministic eval checks (unit-tested in tests/unit/test_eval_checks.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

from research_agent.citations import extract_citation_ids, normalize_url
from research_agent.domain import Report, Source


class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    rationale: str


@dataclass(frozen=True)
class CheckResult:
    schema_valid: bool
    citations_resolve: bool
    unresolved: list[int]
    distinct_domains: int
    min_sources_ok: bool


def _domain(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.")


def run_checks(
    report_json: dict[str, Any], run_sources: list[Source], min_sources: int
) -> CheckResult:
    """schema valid, every [n] maps to a source retrieved in the run, enough distinct domains."""
    try:
        report = Report.model_validate(report_json)
    except ValidationError:
        return CheckResult(False, False, [], 0, False)
    retrieved = {normalize_url(s.url) for s in run_sources}
    by_id = {s.id: s for s in report.sources}
    unresolved = [
        i
        for i in extract_citation_ids(report.markdown)
        if i not in by_id or normalize_url(by_id[i].url) not in retrieved
    ]
    domains = {_domain(s.url) for s in report.sources}
    return CheckResult(
        schema_valid=True,
        citations_resolve=not unresolved and not report.invalid_citations,
        unresolved=unresolved + report.invalid_citations,
        distinct_domains=len(domains),
        min_sources_ok=len(domains) >= min_sources,
    )
