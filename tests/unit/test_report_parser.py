from research_agent.domain import Source
from research_agent.report import build_report, parse_report_markdown

MD = """## Summary
Observability for LLM apps relies on tracing and evals [1].

## Key Findings
- OpenTelemetry GenAI conventions standardize spans [1].
- Evals catch regressions [2].

## Tracing
Tracing tools capture prompts and tokens [1][9].

## Evaluation
Offline and online evals [2].

## Open Questions
- Cost attribution across agents.
"""


def test_parse_sections() -> None:
    parsed = parse_report_markdown(MD)
    assert parsed.summary.startswith("Observability")
    assert parsed.key_findings == [
        "OpenTelemetry GenAI conventions standardize spans [1].",
        "Evals catch regressions [2].",
    ]
    assert [s.title for s in parsed.sections] == ["Tracing", "Evaluation"]
    assert parsed.open_questions == ["Cost attribution across agents."]


def test_parse_is_lenient_without_headings() -> None:
    parsed = parse_report_markdown("Just a paragraph [1].\n\nAnother one.")
    assert parsed.summary == "Just a paragraph [1]."
    assert parsed.sections[0].title == "Details"


def test_build_report_maps_citations_and_drops_invalid_ones() -> None:
    sources = [
        Source(id=1, url="https://otel.example", title="OTel", excerpt="x", fetched=True),
        Source(id=2, url="https://evals.example", title="Evals", excerpt="y"),
        Source(id=3, url="https://unused.example", title="Unused", excerpt="z"),
        Source(id=10, url="https://ten.example", title="Ten", excerpt="t"),
    ]
    report = build_report(
        question="q", markdown=MD, sources=sources, failed_sub_questions=["sq3: x"]
    )
    assert [s.id for s in report.sources] == [1, 2]
    assert report.invalid_citations == [9]
    assert "[9]" not in report.markdown
    assert "## Sources" in report.markdown
    assert "[1] [OTel](https://otel.example)" in report.markdown
    assert report.failed_sub_questions == ["sq3: x"]
    assert "[9]" not in report.sections[0].content
