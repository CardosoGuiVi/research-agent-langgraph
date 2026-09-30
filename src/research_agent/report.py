"""Turn the streamed markdown answer into a validated `Report`.

The answer node streams markdown (good UX: tokens appear as they are generated) that follows a
fixed heading contract from the answer prompt. This module parses it into the typed schema,
validates citations against the run's sources and appends the sources list. Parsing is lenient:
a model that ignores the headings still yields a valid report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from research_agent.citations import extract_citation_ids, sanitize_citations
from research_agent.domain import Report, ReportSection, Source, SourceRef

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


@dataclass
class ParsedMarkdown:
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    sections: list[ReportSection] = field(default_factory=list)


def _bullets(block: str) -> list[str]:
    items = [m.group(1).strip() for line in block.splitlines() if (m := _BULLET_RE.match(line))]
    return items or [p.strip() for p in block.split("\n\n") if p.strip()]


def parse_report_markdown(markdown: str) -> ParsedMarkdown:
    parsed = ParsedMarkdown()
    matches = list(_HEADING_RE.finditer(markdown))
    preamble = markdown[: matches[0].start()] if matches else markdown
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        title, body = m.group(1).strip(), markdown[m.end() : end].strip()
        key = title.lower().rstrip(":")
        if key == "summary":
            parsed.summary = body
        elif key in {"key findings", "findings"}:
            parsed.key_findings = _bullets(body)
        elif key in {"open questions", "limitations"}:
            parsed.open_questions = _bullets(body)
        elif key in {"sources", "references"}:
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
    *, question: str, markdown: str, sources: list[Source], failed_sub_questions: list[str]
) -> Report:
    valid_ids = {s.id for s in sources}
    clean_md, invalid = sanitize_citations(markdown.strip(), valid_ids)
    cited = set(extract_citation_ids(clean_md, max_id=max(valid_ids, default=0)))
    refs = [
        SourceRef(id=s.id, url=s.url, title=s.title, fetched=s.fetched)
        for s in sources
        if s.id in cited
    ]
    parsed = parse_report_markdown(clean_md)
    full_md = clean_md
    if refs:
        full_md += "\n\n## Sources\n" + render_sources(refs)
    return Report(
        question=question,
        summary=parsed.summary,
        sections=parsed.sections,
        key_findings=parsed.key_findings,
        open_questions=parsed.open_questions,
        sources=refs,
        failed_sub_questions=failed_sub_questions,
        invalid_citations=invalid,
        markdown=full_md,
    )
