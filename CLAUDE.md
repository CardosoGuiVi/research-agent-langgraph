# CLAUDE.md

Research agent: FastAPI -> LangGraph StateGraph -> Claude (langchain-anthropic). Public portfolio repo.

## Commands
- `make install` — uv sync (dev group) + pre-commit hooks
- `make lint` / `make format` — ruff
- `make typecheck` — mypy --strict on `src/`
- `make test` — unit + API tests, coverage gate 85% (never hits the network)
- `make test-live` — `@pytest.mark.live` tests (real APIs, costs money)
- `make test-e2e` — Playwright smoke test (`@pytest.mark.e2e`, needs `make up`)
- `make eval` — live mini-eval, results in `evals/results/` (gitignored)
- `make up` / `down` / `logs` — docker compose
- `make studio` — LangGraph Studio (`langgraph dev`)
- Single test: `uv run pytest tests/unit/test_config.py::test_name -q`

## Architecture
Graph: START -> planner -> research (one parallel branch per sub-question via `Send`) ->
analysis -> (research again for gaps | answer) -> END. Iteration cap enforced in `graph/routing.py`.

- `config.py` — pydantic-settings; secrets are `SecretStr`.
- `logging.py` — structlog JSON + secret redaction processor.
- `domain.py` — Pydantic models (plan, notes, sources, assessments, report).
- `llm/` — `LLM` Protocol (structured / tool_step / stream_text), Anthropic adapter, pricing.
- `tools/` — `SearchProvider` (Tavily, DuckDuckGo), per-run budget + cache, SSRF-safe fetcher,
  retry/backoff helper, error hierarchy (`TransientError` = retryable).
- `graph/state.py` — typed state; `notes`/`sources`/`questions` accumulate across runs on a
  thread (checkpointer memory); the `merge_sources` reducer assigns citation ids at fan-in.
- `graph/context.py` — `GraphDeps` (process-wide adapters) and `RunScope` (per-run budget,
  cache, metrics) passed as LangGraph runtime `context`.
- `graph/nodes/` — planner, research, analysis, answer. Nodes never raise for expected failures;
  they record errors and degrade (see tests in `tests/unit/graph/`).
- `citations.py` / `report.py` — [n] parsing/validation; streamed markdown -> `Report`.
- `prompts/*.md` — versioned prompts (front matter `version`).
- `container.py` — composition root (wires adapters; tests build it with fakes).
- `api/` — FastAPI app factory, routes, schemas, middleware.

## Conventions
- TDD: failing test first. Tests are fast and deterministic; no network in the default suite.
- LLM, search and page fetching live behind Protocols; tests use fakes from `tests/fakes.py`
  (`FakeLLM` is scripted per schema; `default_research_policy` drives search -> fetch -> notes).
- Node signature is `(state, runtime)`; mypy matches LangGraph's protocol by parameter name.
- Anthropic SDK uses `httpx2`, so respx cannot mock it; adapter tests stub ChatAnthropic methods.
- Prompts live in `src/research_agent/prompts/*.md` (versioned), never inline.
- Conventional Commits, one per green milestone. Never push.
- Dependencies pinned with `==`; `uv.lock` is committed.

## Guardrails
- Never read, create or edit `.env` (denied in `.claude/settings.json`). Maintain `.env.example` only.
- Never log secrets; the redaction processor is a safety net, not a licence.
- Run `git status` before committing; gitleaks runs in pre-commit and CI.
