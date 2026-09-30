"""AnthropicLLM with ChatAnthropic's transport methods stubbed (no network)."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel

from research_agent.llm.anthropic import AnthropicLLM
from research_agent.llm.base import LLMError, LLMRefusalError, Tier, Usage
from tests.fakes import make_settings

USAGE = {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}


class Answer(BaseModel):
    value: int


def _stub_generate(monkeypatch: pytest.MonkeyPatch, message: AIMessage) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake(
        self: ChatAnthropic,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        seen.append(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "kwargs": kwargs,
                "betas": self.betas,
                "model_kwargs": self.model_kwargs,
                "effort": self.reasoning_effort,
                "messages": messages,
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatAnthropic, "_agenerate", fake)
    return seen


def _llm(**overrides: Any) -> AnthropicLLM:
    return AnthropicLLM(make_settings(**overrides))


def test_requires_api_key() -> None:
    with pytest.raises(LLMError):
        AnthropicLLM(make_settings(anthropic_api_key=None))


async def test_structured_parses_json_and_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    msg = AIMessage(
        content='{"value": 42}',
        usage_metadata=USAGE,
        response_metadata={"model_name": "claude-haiku-4-5", "stop_reason": "end_turn"},
    )
    seen = _stub_generate(monkeypatch, msg)
    out, usage = await _llm().structured(
        Answer, system="s", user="u", tier=Tier.FAST, max_tokens=123
    )
    assert out == Answer(value=42)
    assert usage == Usage("claude-haiku-4-5", 12, 7)
    assert seen[0]["max_tokens"] == 123
    assert seen[0]["model"] == "claude-haiku-4-5"
    assert seen[0]["kwargs"]["output_config"]["format"]["type"] == "json_schema"


async def test_structured_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generate(monkeypatch, AIMessage(content="not json", usage_metadata=USAGE))
    with pytest.raises(LLMError):
        await _llm().structured(Answer, system="s", user="u", tier=Tier.FAST, max_tokens=10)


async def test_structured_refusal_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generate(
        monkeypatch,
        AIMessage(content="", usage_metadata=USAGE, response_metadata={"stop_reason": "refusal"}),
    )
    with pytest.raises(LLMRefusalError):
        await _llm().structured(Answer, system="s", user="u", tier=Tier.FAST, max_tokens=10)


async def test_smart_tier_uses_effort_and_refusal_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_generate(monkeypatch, AIMessage(content='{"value": 1}', usage_metadata=USAGE))
    await _llm(llm_model_smart="claude-opus-5", llm_smart_effort="low").structured(
        Answer, system="s", user="u", tier=Tier.SMART, max_tokens=10
    )
    assert seen[0]["effort"] == "low"
    assert seen[0]["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen[0]["model_kwargs"] == {"fallbacks": "default"}


async def test_tool_step_binds_tools_and_returns_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    ai, usage = await _llm().tool_step(
        [HumanMessage("hi")], system="sys", tools=[tool], tier=Tier.FAST, max_tokens=50
    )
    assert ai.tool_calls[0]["name"] == "web_search"
    assert usage.input_tokens == 12
    assert seen[0]["kwargs"]["tools"][0]["name"] == "web_search"
    assert "tool_choice" not in seen[0]["kwargs"]


async def test_tool_step_wrap_up_sends_tool_choice_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_generate(monkeypatch, AIMessage(content="notes", usage_metadata=USAGE))
    tool = {"name": "web_search", "description": "d", "input_schema": {"type": "object"}}
    await _llm().tool_step(
        [HumanMessage("hi")],
        system="sys",
        tools=[tool],
        tier=Tier.FAST,
        max_tokens=50,
        allow_tool_calls=False,
    )
    assert seen[0]["kwargs"]["tool_choice"] == {"type": "none"}


async def test_stream_text_yields_text_then_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream(
        self: ChatAnthropic,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=[{"type": "thinking", "thinking": "", "index": 0}])
        )
        yield ChatGenerationChunk(message=AIMessageChunk(content="Hello "))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="world",
                usage_metadata=USAGE,
                response_metadata={"model_name": "claude-opus-5", "stop_reason": "end_turn"},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_stream)
    items = [
        i async for i in _llm().stream_text(system="s", user="u", tier=Tier.SMART, max_tokens=100)
    ]
    assert items[:-1] == ["Hello ", "world"]
    assert items[-1] == Usage("claude-opus-5", 12, 7)
