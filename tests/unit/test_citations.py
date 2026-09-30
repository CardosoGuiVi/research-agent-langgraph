from research_agent.citations import (
    SourceRegistry,
    extract_citation_ids,
    replace_urls_with_citations,
    sanitize_citations,
)
from research_agent.domain import Source


def _src(i: int, url: str) -> Source:
    return Source(id=i, url=url, title=f"T{i}", excerpt="e")


def test_registry_assigns_stable_ids_and_dedupes_urls() -> None:
    reg = SourceRegistry()
    a = reg.add(url="https://a.example/x", title="A", excerpt="a")
    b = reg.add(url="https://b.example", title="B", excerpt="b")
    a_again = reg.add(url="https://a.example/x/", title="A2", excerpt="longer excerpt a")
    assert (a.id, b.id, a_again.id) == (1, 2, 1)
    assert [s.id for s in reg.sources] == [1, 2]
    # A fetched page upgrades the stored excerpt/flag of an already-known source.
    reg.add(url="https://b.example", title="B", excerpt="full text", fetched=True)
    assert reg.get(2) is not None
    assert reg.get(2).fetched  # type: ignore[union-attr]


def test_registry_normalizes_fragments_and_tracking_params() -> None:
    reg = SourceRegistry()
    one = reg.add(url="https://a.example/p?utm_source=x#section", title="A", excerpt="")
    two = reg.add(url="https://A.example/p", title="A", excerpt="")
    assert one.id == two.id


def test_extract_citation_ids_handles_grouped_and_ranged_forms() -> None:
    text = "Tracing helps [1]. Evals too [2][3]. Also [4, 5] and [6-7]. Not a cite: [x] [2020]."
    assert extract_citation_ids(text) == [1, 2, 3, 4, 5, 6, 7]


def test_sanitize_removes_unknown_citations_and_reports_them() -> None:
    text = "A [1]. B [9]. C [2][9]."
    clean, invalid = sanitize_citations(text, valid_ids={1, 2, 10})
    assert clean == "A [1]. B. C [2]."
    assert invalid == [9]


def test_replace_urls_with_citations() -> None:
    sources = [_src(1, "https://a.example/x"), _src(2, "https://b.example")]
    notes = "- Fact one (https://a.example/x)\n- Fact two (https://b.example/)\n- Other (https://c.example)"
    out = replace_urls_with_citations(notes, sources)
    assert "- Fact one [1]" in out
    assert "- Fact two [2]" in out
    assert "https://c.example" in out


def test_bracketed_numbers_beyond_the_source_range_are_literal_text() -> None:
    # Found by the eval: "[429, 500, 503]" (HTTP status codes) is not a citation.
    text = "Retry on [429, 500, 503] with backoff [1]. See [2]."
    clean, invalid = sanitize_citations(text, valid_ids={1, 2, 3})
    assert clean == text
    assert invalid == []
    assert extract_citation_ids(text, max_id=3) == [1, 2]


def test_out_of_list_id_within_range_is_still_removed() -> None:
    clean, invalid = sanitize_citations("A [2]. B [3].", valid_ids={1, 3})
    assert clean == "A. B [3]."
    assert invalid == [2]
