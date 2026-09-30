"""Citation bookkeeping: stable source numbering, [n] parsing and validation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from research_agent.domain import Source

# [1] / [1, 2] / [3-5]; 1-3 digits so years like [2020] are not treated as citations.
_CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*)\]")
_TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|ref|ref_src)$", re.I)
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def normalize_url(url: str) -> str:
    """Canonical form used to dedupe sources (drops fragment, tracking params, trailing /)."""
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)])
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


class SourceRegistry:
    """Assigns citation numbers to URLs in first-seen order; stable across a thread."""

    def __init__(self, sources: Iterable[Source] = ()) -> None:
        self._by_key: dict[str, Source] = {}
        for src in sources:
            self._by_key[normalize_url(src.url)] = src

    @property
    def sources(self) -> list[Source]:
        return sorted(self._by_key.values(), key=lambda s: s.id)

    def get(self, source_id: int) -> Source | None:
        return next((s for s in self._by_key.values() if s.id == source_id), None)

    def add(self, *, url: str, title: str, excerpt: str, fetched: bool = False) -> Source:
        key = normalize_url(url)
        existing = self._by_key.get(key)
        if existing is None:
            src = Source(
                id=len(self._by_key) + 1, url=url, title=title, excerpt=excerpt, fetched=fetched
            )
            self._by_key[key] = src
            return src
        if (fetched and not existing.fetched) or len(excerpt) > len(existing.excerpt):
            existing = existing.model_copy(
                update={
                    "excerpt": excerpt
                    if len(excerpt) > len(existing.excerpt)
                    else existing.excerpt,
                    "fetched": existing.fetched or fetched,
                    "title": existing.title or title,
                }
            )
            self._by_key[key] = existing
        return existing

    def id_for(self, url: str) -> int | None:
        src = self._by_key.get(normalize_url(url))
        return src.id if src else None


def _expand(group: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if m := re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part):
            lo, hi = int(m.group(1)), int(m.group(2))
            ids.extend(range(lo, hi + 1) if 0 < hi - lo < 50 else [lo, hi])
        else:
            ids.append(int(part))
    return ids


def extract_citation_ids(text: str) -> list[int]:
    """Distinct citation numbers in order of first appearance."""
    seen: dict[int, None] = {}
    for match in _CITATION_RE.finditer(text):
        for i in _expand(match.group(1)):
            seen.setdefault(i, None)
    return list(seen)


def sanitize_citations(text: str, valid_ids: set[int]) -> tuple[str, list[int]]:
    """Drop citation numbers that map to no source. Returns (clean_text, invalid_ids)."""
    invalid: dict[int, None] = {}

    def repl(match: re.Match[str]) -> str:
        ids = _expand(match.group(1))
        kept = [i for i in ids if i in valid_ids]
        for i in ids:
            if i not in valid_ids:
                invalid.setdefault(i, None)
        return "".join(f"[{i}]" for i in kept)

    clean = _CITATION_RE.sub(repl, text)
    clean = re.sub(r"[ \t]+([.,;:])", r"\1", clean)  # "claim [9]." -> "claim ."-> "claim."
    return clean, list(invalid)


def replace_urls_with_citations(text: str, sources: Iterable[Source]) -> str:
    """Rewrite "(https://...)" references in research notes as "[n]"."""
    index = {normalize_url(s.url): s.id for s in sources}

    def repl(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;:")
        tail = match.group(0)[len(url) :]
        sid = index.get(normalize_url(url))
        return f"[{sid}]{tail}" if sid is not None else match.group(0)

    out = _URL_RE.sub(repl, text)
    return re.sub(r"\((\[\d+\](?:\s*,?\s*\[\d+\])*)\)", r"\1", out)  # "([1])" -> "[1]"
