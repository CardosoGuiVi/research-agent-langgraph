.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test test-live test-e2e eval graph up down logs studio check

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install deps (incl. dev) and git hooks
	$(UV) sync --frozen
	$(UV) run pre-commit install

lint: ## Lint and check formatting (no changes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-fix lint issues and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy --strict on src/
	$(UV) run mypy

test: ## Unit + API tests with coverage gate (no network, no paid APIs)
	$(UV) run pytest --cov --cov-report=term-missing

test-live: ## Tests that hit real APIs (costs money; needs .env)
	$(UV) run pytest -m live -v

test-e2e: ## Playwright smoke test against a running stack (make up first)
	$(UV) sync --frozen --group e2e
	$(UV) run playwright install chromium
	$(UV) run pytest -m e2e -v

eval: ## Mini eval with LLM-as-judge (live, costs money)
	$(UV) run python -m evals.run

graph: ## Export the graph as Mermaid to docs/graph.mmd
	$(UV) run python -m research_agent.graph.export docs/graph.mmd

studio: ## Open the graph in LangGraph Studio (langgraph dev)
	$(UV) run --group studio langgraph dev

check: lint typecheck test ## Everything CI runs locally

up: ## Build and start the stack
	docker compose up --build -d

down: ## Stop the stack
	docker compose down

logs: ## Tail service logs
	docker compose logs -f
