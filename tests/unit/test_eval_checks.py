from evals.checks import run_checks
from research_agent.domain import Source
from research_agent.report import build_report

SOURCES = [
    Source(id=1, url="https://www.a.example/x", title="A", fetched=True),
    Source(id=2, url="https://b.example/y", title="B"),
    Source(id=3, url="https://a.example/z", title="A2"),
]


def _report(md: str) -> dict:  # type: ignore[type-arg]
    return build_report(
        question="q", markdown=md, sources=SOURCES, failed_sub_questions=[]
    ).model_dump()


def test_valid_report_passes() -> None:
    result = run_checks(_report("## Summary\nX [1] and Y [2] and Z [3]."), SOURCES, min_sources=2)
    assert result.schema_valid
    assert result.citations_resolve
    assert result.distinct_domains == 2  # a.example counted once
    assert result.min_sources_ok


def test_citation_to_source_not_retrieved_fails() -> None:
    result = run_checks(_report("## Summary\nX [1] [2]."), SOURCES[:1], min_sources=1)
    assert not result.citations_resolve
    assert result.unresolved == [2]


def test_invalid_citation_and_too_few_sources() -> None:
    result = run_checks(_report("## Summary\nX [1] [9]."), SOURCES, min_sources=3)
    assert not result.citations_resolve
    assert 9 in result.unresolved
    assert not result.min_sources_ok


def test_schema_invalid() -> None:
    result = run_checks({"question": "q"}, SOURCES, min_sources=1)
    assert not result.schema_valid
