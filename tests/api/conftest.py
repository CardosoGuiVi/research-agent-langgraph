import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from research_agent.api.app import create_app
from research_agent.container import Container
from research_agent.domain import (
    AnalysisOutput,
    PlannedSubQuestion,
    PlannerOutput,
    SubQuestionAssessment,
)
from tests.fakes import FakeLLM, build_test_container

PLAN = PlannerOutput(
    sub_questions=[
        PlannedSubQuestion(question="Tracing tools?", search_queries=["llm tracing"], rationale=""),
        PlannedSubQuestion(question="Eval tools?", search_queries=["llm evals"], rationale=""),
    ],
    strategy="s",
)
ANALYSIS = AnalysisOutput(
    assessments=[
        SubQuestionAssessment(sub_question_id="sq1", coverage="sufficient"),
        SubQuestionAssessment(sub_question_id="sq2", coverage="sufficient"),
    ]
)


def fake_llm(**kw: Any) -> FakeLLM:
    return FakeLLM(structured={PlannerOutput: PLAN, AnalysisOutput: ANALYSIS}, **kw)


@pytest.fixture
def container() -> Container:
    return build_test_container(llm=fake_llm())


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    for block in body.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if data is not None:
            events.append((name, data))
    return events
