# 4. Stream the answer as markdown and parse it into the report schema

- Status: accepted
- Date: 2026-09-30

## Context and problem statement

The report must be structured (summary, sections, key findings, open questions, sources) *and*
streamed token by token to the UI.

## Considered options

1. Structured output (JSON schema) from the model, streamed as partial JSON.
2. Markdown with a fixed heading contract, streamed, then parsed into the `Report` schema.
3. Stream markdown, then a second LLM call to convert it into JSON.

## Decision outcome

**Option 2.** The answer prompt fixes the headings; `report.py` parses them leniently (a model
that ignores the contract still yields a valid report) and `citations.py` validates every `[n]`
against the sources that were shown to the model. Unknown in-range ids are removed and reported
in `invalid_citations`; bracketed numbers outside the source range (e.g. `[429, 503]`) are left
as literal text. That rule came from the eval, which caught HTTP status codes being treated as
citations.

### Consequences

- Good: readable streaming, one LLM call, deterministic and unit-tested structure extraction.
- Good: citation numbers in the streamed text and the final report are identical (no renumbering).
- Bad: parsing is heuristic. Mitigated by lenient parsing, schema validation in the eval and a
  fallback that never fails the run.
