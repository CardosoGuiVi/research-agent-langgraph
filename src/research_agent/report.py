"""Turn the streamed markdown answer into a validated `Report`.

The answer node streams markdown (good UX: tokens appear as they are generated) that follows a
fixed heading contract from the answer prompt. This module parses it into the typed schema,
validates citations against the run's sources and appends the sources list. Parsing is lenient:
a model that ignores the headings still yields a valid report.

Headings are localized: the planner detects the question's language, the answer prompt is given
the exact headings for it, and the parser matches those (plus the English ones, in case the model
ignores the instruction). Languages without a table get English headings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from research_agent.citations import extract_citation_ids, sanitize_citations
from research_agent.domain import Report, ReportSection, Source, SourceRef

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


@dataclass(frozen=True)
class Headings:
    summary: str
    key_findings: str
    open_questions: str
    sources: str
    # Text for reports built without the model (every sub-question failed / synthesis failed).
    research_failed: str
    not_researched: str
    synthesis_failed: str


HEADINGS: dict[str, Headings] = {
    "en": Headings(
        summary="Summary",
        key_findings="Key Findings",
        open_questions="Open Questions",
        sources="Sources",
        research_failed="Research could not be completed: {reason}. No sources were found, "
        "so no answer is given.",
        not_researched="Not researched: {question}",
        synthesis_failed="The synthesis step failed ({error}), so this report lists the raw "
        "research notes instead.",
    ),
    "pt": Headings(
        summary="Resumo",
        key_findings="Principais conclusões",
        open_questions="Questões em aberto",
        sources="Fontes",
        research_failed="Não foi possível concluir a pesquisa: {reason}. Nenhuma fonte foi "
        "encontrada, então nenhuma resposta é dada.",
        not_researched="Não pesquisado: {question}",
        synthesis_failed="A etapa de síntese falhou ({error}), então este relatório lista as "
        "notas de pesquisa brutas.",
    ),
}
_LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese"}
_EN = HEADINGS["en"]


def _base_language(code: str) -> str:
    return code.strip().lower().replace("_", "-").split("-")[0] or "en"


def headings_for(language: str) -> Headings:
    return HEADINGS.get(_base_language(language), _EN)


def language_name(language: str) -> str:
    code = _base_language(language)
    return _LANGUAGE_NAMES.get(code, f"the language with ISO 639-1 code '{code}'")


@dataclass
class ParsedMarkdown:
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    sections: list[ReportSection] = field(default_factory=list)


def _bullets(block: str) -> list[str]:
    items = [m.group(1).strip() for line in block.splitlines() if (m := _BULLET_RE.match(line))]
    return items or [p.strip() for p in block.split("\n\n") if p.strip()]


def _keys(*names: str) -> set[str]:
    return {n.lower() for n in names}


def parse_report_markdown(markdown: str, headings: Headings = _EN) -> ParsedMarkdown:
    summary_keys = _keys("summary", headings.summary)
    findings_keys = _keys("key findings", "findings", headings.key_findings)
    open_keys = _keys("open questions", "limitations", headings.open_questions)
    sources_keys = _keys("sources", "references", headings.sources)
    parsed = ParsedMarkdown()
    matches = list(_HEADING_RE.finditer(markdown))
    preamble = markdown[: matches[0].start()] if matches else markdown
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        title, body = m.group(1).strip(), markdown[m.end() : end].strip()
        key = title.lower().rstrip(":")
        if key in summary_keys:
            parsed.summary = body
        elif key in findings_keys:
            parsed.key_findings = _bullets(body)
        elif key in open_keys:
            parsed.open_questions = _bullets(body)
        elif key in sources_keys:
            continue  # we append our own authoritative list
        else:
            parsed.sections.append(ReportSection(title=title, content=body))
    paragraphs = [p.strip() for p in preamble.split("\n\n") if p.strip()]
    if not parsed.summary and paragraphs:
        parsed.summary = paragraphs[0]
        paragraphs = paragraphs[1:]
    if paragraphs:
        parsed.sections.insert(0, ReportSection(title="Details", content="\n\n".join(paragraphs)))
    return parsed


def render_sources(sources: Iterable[SourceRef]) -> str:
    return "\n".join(f"[{s.id}] [{s.title}]({s.url})" for s in sources)


def build_report(
    *,
    question: str,
    markdown: str,
    sources: list[Source],
    failed_sub_questions: list[str],
    language: str = "en",
) -> Report:
    headings = headings_for(language)
    valid_ids = {s.id for s in sources}
    clean_md, invalid = sanitize_citations(markdown.strip(), valid_ids)
    cited = set(extract_citation_ids(clean_md, max_id=max(valid_ids, default=0)))
    refs = [
        SourceRef(id=s.id, url=s.url, title=s.title, fetched=s.fetched)
        for s in sources
        if s.id in cited
    ]
    parsed = parse_report_markdown(clean_md, headings)
    full_md = clean_md
    if refs:
        full_md += f"\n\n## {headings.sources}\n" + render_sources(refs)
    return Report(
        question=question,
        language=_base_language(language),
        summary=parsed.summary,
        sections=parsed.sections,
        key_findings=parsed.key_findings,
        open_questions=parsed.open_questions,
        sources=refs,
        failed_sub_questions=failed_sub_questions,
        invalid_citations=invalid,
        markdown=full_md,
    )
