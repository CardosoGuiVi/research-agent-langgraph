# 2. Parallel research per sub-question with a bounded refinement loop

- Status: accepted
- Date: 2026-09-30

## Context and problem statement

Research quality comes from covering several angles; latency comes mostly from sequential LLM
and HTTP round trips. We need parallelism without losing determinism, and a way to fix gaps
without letting the agent loop forever or overspend.

## Decision outcome

- The planner (fast model, structured output) produces 3-6 sub-questions with 1-3 queries each.
- `route_after_planner` returns one `Send("research", task)` per sub-question. Each branch runs
  a bounded tool-calling loop (`max_research_steps`) with `web_search` and `fetch_page`; tool
  calls issued in the same turn execute concurrently.
- Branches never raise: failures (search down, no results, LLM error) are recorded on the
  branch's `ResearchNote`, so one failing sub-question degrades the report instead of the run.
- Branches return sources with placeholder ids; the `merge_sources` reducer assigns citation
  numbers in first-seen order at fan-in, where LangGraph applies updates sequentially. Numbers
  are therefore stable across runs on a thread.
- `analysis` (fast model) rates coverage per sub-question and proposes refined queries.
  `route_after_analysis` sends only sub-questions with gaps and *new* queries back to research,
  and enforces `max_iterations` in code regardless of what the model says. When no further round
  is possible the analysis LLM call is skipped entirely.
- Run-level limits (searches, fetches, tokens) live in a `RunBudget` shared by all branches via
  runtime context; repeated queries are served from a per-run cache.

### Consequences

- Good: wall-clock time is roughly one branch, not the sum; the loop is bounded and testable.
- Good: the search cap is exact even with parallel branches (single event loop, synchronous
  check-and-increment).
- Bad: branches do not share findings mid-flight, so two branches can fetch the same page.
  The per-run search cache limits the waste for identical queries.
- Bad: the research loop re-sends fetched page text on each step, which dominates token cost
  (see README results). Next step: prompt caching and summarizing tool results.
