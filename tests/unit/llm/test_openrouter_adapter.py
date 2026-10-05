"""OpenRouterProvider with ChatOpenAI's transport methods stubbed (no network, no real key)."""

from collections.abc import AsyncIterator
from typing import Any, NoReturn

import openai._base_client
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from research_agent.llm.base import LLMError, LLMRefusalError, Tier, Usage
from research_agent.llm.openrouter import OpenRouterProvider
from tests.fakes import make_settings

USAGE = {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
FAST = "vendor/fast-model"
SMART = "vendor/smart-model"


class Answer(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything reaches the OpenAI SDK's HTTP layer (httpx2, so no respx)."""

    def blocked(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("network call to OpenRouter attempted in a unit test")

    monkeypatch.setattr(openai._base_client.AsyncAPIClient, "request", blocked)
    monkeypatch.setattr(openai._base_client.SyncAPIClient, "request", blocked)


def _stub_generate(monkeypatch: pytest.MonkeyPatch, message: AIMessage) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake(
        self: ChatOpenAI,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        seen.append(
            {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "base_url": self.openai_api_base,
                "kwargs": kwargs,
                "messages": messages,
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake)
    return seen


def _provider(**overrides: Any) -> OpenRouterProvider:
    base: dict[str, Any] = {
        "llm_provider": "openrouter",
        "anthropic_api_key": None,
        "openrouter_api_key": "or-test",
        "openrouter_model": FAST,
    }
    base.update(overrides)
    return OpenRouterProvider(make_settings(**base))


def test_instantiates_without_network() -> None:
    provider = _provider()
    assert isinstance(provider, OpenRouterProvider)


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        _provider(openrouter_api_key=None)


def test_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_MODEL"):
        _provider(openrouter_model=None)


async def test_structured_uses_json_schema_and_reports_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg = AIMessage(
        content='{"value": 42}',
        usage_metadata=USAGE,
        response_metadata={"model_name": FAST, "finish_reason": "stop"},
    )
    seen = _stub_generate(monkeypatch, msg)
    out, usage = await _provider().structured(
        Answer, system="s", user="u", tier=Tier.FAST, max_tokens=123
    )
    assert out == Answer(value=42)
    assert usage == Usage(FAST, 12, 7)
    assert seen[0]["model"] == FAST
    assert seen[0]["max_tokens"] == 123
    assert seen[0]["base_url"] == "https://openrouter.ai/api/v1"
    # A JSON-schema dict (not the class), so ChatOpenAI parses the content instead of
    # requiring the SDK's strict `parse` path.
    fmt = seen[0]["kwargs"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Answer"
    assert fmt["json_schema"]["schema"]["properties"] == Answer.model_json_schema()["properties"]
    assert "tools" not in seen[0]["kwargs"]


async def test_structured_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generate(monkeypatch, AIMessage(content="not json", usage_metadata=USAGE))
    with pytest.raises(LLMError):
        await _provider().structured(Answer, system="s", user="u", tier=Tier.FAST, max_tokens=10)


async def test_structured_json_not_matching_schema_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generate(monkeypatch, AIMessage(content='{"value": "many"}', usage_metadata=USAGE))
    with pytest.raises(LLMError, match="Answer"):
        await _provider().structured(Answer, system="s", user="u", tier=Tier.FAST, max_tokens=10)


async def test_content_filter_is_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generate(
        monkeypatch,
        AIMessage(
            content="", usage_metadata=USAGE, response_metadata={"finish_reason": "content_filter"}
        ),
    )
    with pytest.raises(LLMRefusalError):
        await _provider().tool_step(
            [HumanMessage("hi")], system="s", tools=[], tier=Tier.FAST, max_tokens=10
        )


async def test_smart_tier_uses_smart_model_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_generate(monkeypatch, AIMessage(content="ok", usage_metadata=USAGE))
    await _provider(openrouter_model_smart=SMART).tool_step(
        [HumanMessage("hi")], system="s", tools=[], tier=Tier.SMART, max_tokens=10
    )
    await _provider().tool_step(
        [HumanMessage("hi")], system="s", tools=[], tier=Tier.SMART, max_tokens=10
    )
    assert [s["model"] for s in seen] == [SMART, FAST]


async def test_tool_step_converts_anthropic_format_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    msg = AIMessage(
        content="",
        usage_metadata=USAGE,
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "t1"}],
    )
    seen = _stub_generate(monkeypatch, msg)
    tool = {
        "name": "web_search",
        "description": "d",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    ai, usage = await _provider().tool_step(
        [HumanMessage("hi")], system="sys", tools=[tool], tier=Tier.FAST, max_tokens=50
    )
    assert ai.tool_calls[0]["name"] == "web_search"
    assert usage.input_tokens == 12
    sent = seen[0]["kwargs"]["tools"][0]
    assert sent["type"] == "function"
    assert sent["function"]["name"] == "web_search"
    assert sent["function"]["parameters"] == tool["input_schema"]
    assert "tool_choice" not in seen[0]["kwargs"]


async def test_tool_step_wrap_up_sends_tool_choice_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_generate(monkeypatch, AIMessage(content="notes", usage_metadata=USAGE))
    tool = {"name": "web_search", "description": "d", "input_schema": {"type": "object"}}
    await _provider().tool_step(
        [HumanMessage("hi")],
        system="sys",
        tools=[tool],
        tier=Tier.FAST,
        max_tokens=50,
        allow_tool_calls=False,
    )
    assert seen[0]["kwargs"]["tool_choice"] == "none"


async def test_stream_text_yields_text_then_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream(
        self: ChatOpenAI,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # OpenAI-style streams repeat model_name on every chunk; merging must not concatenate it.
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="Hello ", response_metadata={"model_name": SMART})
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="world",
                usage_metadata=USAGE,
                response_metadata={"model_name": SMART, "finish_reason": "stop"},
            )
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", fake_stream)
    provider = _provider(openrouter_model_smart=SMART)
    items = [
        i async for i in provider.stream_text(system="s", user="u", tier=Tier.SMART, max_tokens=9)
    ]
    assert items[:-1] == ["Hello ", "world"]
    assert items[-1] == Usage(SMART, 12, 7)
