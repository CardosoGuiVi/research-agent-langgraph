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
(Filled in as milestones land; see README for the full picture.)
- `config.py` — pydantic-settings; secrets are `SecretStr`.
- `logging.py` — structlog JSON + secret redaction processor.
- `container.py` — composition root (wires adapters; tests build it with fakes).
- `api/` — FastAPI app factory, routes, schemas, middleware.

## Conventions
- TDD: failing test first. Tests are fast and deterministic; no network in the default suite.
- LLM, search and page fetching live behind Protocols; tests use fakes from `tests/fakes.py`.
- Prompts live in `src/research_agent/prompts/*.md` (versioned), never inline.
- Conventional Commits, one per green milestone. Never push.
- Dependencies pinned with `==`; `uv.lock` is committed.

## Guardrails
- Never read, create or edit `.env` (denied in `.claude/settings.json`). Maintain `.env.example` only.
- Never log secrets; the redaction processor is a safety net, not a licence.
- Run `git status` before committing; gitleaks runs in pre-commit and CI.
