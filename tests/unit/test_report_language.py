from research_agent.domain import Source
from research_agent.report import build_report, headings_for, language_name, parse_report_markdown

PT = """## Resumo
Observabilidade combina tracing e avaliação [1].

## Principais conclusões
- Convenções GenAI do OpenTelemetry padronizam spans [1].

## Tracing distribuído
Detalhes [1].

## Questões em aberto
- Atribuição de custo entre agentes.
"""


def test_headings_for_supported_and_unsupported_languages() -> None:
    assert headings_for("pt").summary == "Resumo"
    assert headings_for("PT-br").summary == "Resumo"
    assert headings_for("en").summary == "Summary"
    assert headings_for("de").summary == "Summary"  # no table yet: English headings
    assert language_name("pt") == "Portuguese"
    assert language_name("de") == "the language with ISO 639-1 code 'de'"


def test_parse_portuguese_headings() -> None:
    parsed = parse_report_markdown(PT, headings_for("pt"))
    assert parsed.summary.startswith("Observabilidade")
    assert parsed.key_findings == ["Convenções GenAI do OpenTelemetry padronizam spans [1]."]
    assert [s.title for s in parsed.sections] == ["Tracing distribuído"]
    assert parsed.open_questions == ["Atribuição de custo entre agentes."]


def test_english_headings_still_accepted_when_model_ignores_the_language() -> None:
    parsed = parse_report_markdown(
        "## Summary\nX [1].\n\n## Key Findings\n- Y.", headings_for("pt")
    )
    assert parsed.summary == "X [1]."
    assert parsed.key_findings == ["Y."]


def test_build_report_in_portuguese_appends_localized_sources() -> None:
    sources = [Source(id=1, url="https://otel.example", title="OTel")]
    report = build_report(
        question="q", markdown=PT, sources=sources, failed_sub_questions=[], language="pt"
    )
    assert report.language == "pt"
    assert report.summary.startswith("Observabilidade")
    assert "\n## Fontes\n[1] [OTel](https://otel.example)" in report.markdown
