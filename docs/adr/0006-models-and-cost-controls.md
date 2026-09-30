# 6. Two model tiers and hard cost controls

- Status: proposed (review model choice)
- Date: 2026-09-30

## Context and problem statement

Most LLM calls (planning, tool routing in research loops, coverage analysis, judging) are
routine; one call (the final synthesis) determines report quality.

## Decision outcome

- `LLM_MODEL_FAST` (default `claude-haiku-4-5`, $1/$5 per MTok) for planner, research loops,
  analysis and the eval judge.
- `LLM_MODEL_SMART` (default `claude-opus-5`, $5/$25 per MTok) for the answer, with
  `effort=medium` (`LLM_SMART_EFFORT`). For Opus 5 the adapter enables server-side refusal
  fallbacks (`fallbacks: "default"`), so a safety-classifier refusal is retried on a substitute
  model instead of failing the run.
- Every call has `max_tokens`; each run has caps on searches, page fetches, research steps,
  iterations and total tokens. Tokens and estimated cost are logged per run and returned by the API.
- Model IDs come from env and were verified against Anthropic's model list on 2026-09-30.

### Consequences

- Measured cost is about $0.30-0.45 per research question, dominated by fast-model *input*
  tokens in the research loops (fetched page text re-sent each step).
- Cheaper alternatives to evaluate with `make eval`: `LLM_MODEL_SMART=claude-sonnet-5`
  ($2/$10), lower effort, prompt caching on the research loop, smaller `FETCH_MAX_CHARS`.
