"""Browser smoke test for the single-page UI (Playwright).

Run with `make test-e2e`. By default the app is served in-process with fake adapters, so the test
is free and deterministic. Set E2E_BASE_URL (e.g. http://localhost:8000 after `make up`) to run
the same test against the real stack (this calls paid APIs).
"""

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from research_agent.api.app import create_app
from research_agent.domain import (
    AnalysisOutput,
    PlannedSubQuestion,
    PlannerOutput,
    SubQuestionAssessment,
)
from tests.fakes import FakeLLM, build_test_container

pytestmark = pytest.mark.e2e
playwright_api = pytest.importorskip("playwright.sync_api")

ANSWER = """## Summary
Production observability for AI apps combines tracing [1] with evaluation [2].

## Key Findings
- AI systems fail *silently*; OpenTelemetry GenAI conventions standardise spans [1].
- Langfuse offers open-source tracing and evals [2].

## Tracing
Tracing captures prompts, tool calls and token usage [1][2].

## Open Questions
- How to attribute cost across multi-agent systems.
"""


PT_ANSWER = """## Resumo
Observabilidade em produção combina tracing [1] e avaliação [2].

## Principais conclusões
- Convenções GenAI do OpenTelemetry padronizam spans [1].

## Questões em aberto
- Como atribuir custo entre agentes.
"""


def _fake_container(answer: str = ANSWER, language: str = "en"):  # type: ignore[no-untyped-def]
    plan = PlannerOutput(
        sub_questions=[
            PlannedSubQuestion(
                question="Which tracing standards exist?",
                search_queries=["otel genai"],
                rationale="",
            ),
            PlannedSubQuestion(
                question="Which open-source platforms are used?",
                search_queries=["langfuse"],
                rationale="",
            ),
        ],
        strategy="standards, then tools",
        language=language,
    )
    analysis = AnalysisOutput(
        assessments=[
            SubQuestionAssessment(sub_question_id="sq1", coverage="sufficient"),
            SubQuestionAssessment(sub_question_id="sq2", coverage="sufficient"),
        ]
    )
    return build_test_container(
        llm=FakeLLM(structured={PlannerOutput: plan, AnalysisOutput: analysis}, answer=answer)
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve(container):  # type: ignore[no-untyped-def]
    port = _free_port()
    config = uvicorn.Config(create_app(container=container), port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def app_url() -> Iterator[str]:
    if url := os.environ.get("E2E_BASE_URL"):
        yield url.rstrip("/")
        return
    server, thread, url = _serve(_fake_container())
    yield url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def app_url_pt() -> Iterator[str]:
    server, thread, url = _serve(_fake_container(PT_ANSWER, "pt"))
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def test_question_to_cited_report(app_url: str, page) -> None:  # type: ignore[no-untyped-def]
    expect = playwright_api.expect
    page.goto(app_url)
    expect(page.get_by_role("heading", name="Research agent")).to_be_visible()
    page.get_by_label("Question").fill("Analyze the main observability technologies for AI apps.")
    page.get_by_role("button", name="Research").click()

    timeout = 180_000 if os.environ.get("E2E_BASE_URL") else 15_000
    report = page.locator("#report")
    expect(report.get_by_role("heading", name="Summary")).to_be_visible(timeout=timeout)
    expect(page.locator("#trace")).to_contain_text("Planning sub-questions")
    expect(page.locator("#trace")).to_contain_text("Writing the report")
    first_cite = report.locator("a.cite").first
    expect(first_cite).to_be_visible()
    expect(report.locator("ol.sources li").first).to_contain_text("[")
    expect(page.get_by_role("button", name="Ask a follow-up")).to_be_visible()
    expect(report.locator("em", has_text="silently")).to_be_visible()
    if not os.environ.get("E2E_BASE_URL"):
        Path("test-results").mkdir(exist_ok=True)
        page.screenshot(path="test-results/ui.png", full_page=True)


def test_portuguese_report_renders_localized_sources(app_url_pt: str, page) -> None:  # type: ignore[no-untyped-def]
    expect = playwright_api.expect
    page.goto(app_url_pt)
    page.get_by_label("Question").fill(
        "Quais as principais tecnologias de observabilidade para IA?"
    )
    page.get_by_role("button", name="Research").click()
    report = page.locator("#report")
    expect(report.get_by_role("heading", name="Resumo")).to_be_visible(timeout=15_000)
    expect(report.get_by_role("heading", name="Fontes")).to_be_visible()
    expect(report.locator("a.cite").first).to_be_visible()
    expect(report.locator("ol.sources li").first).to_contain_text("[1]")
    expect(report).not_to_contain_text("## Fontes")
