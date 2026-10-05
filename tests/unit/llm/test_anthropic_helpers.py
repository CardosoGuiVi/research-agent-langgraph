import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from research_agent.llm.anthropic import raise_on_refusal, supports_server_side_fallback
from research_agent.llm.base import LLMRefusalError, extract_text, extract_usage


def test_extract_text_skips_thinking_and_tool_blocks() -> None:
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Hello "},
            {"type": "tool_use", "id": "1", "name": "x", "input": {}},
            {"type": "text", "text": "world"},
        ]
    )
    assert extract_text(msg) == "Hello world"


def test_extract_text_from_plain_string_and_chunks() -> None:
    assert extract_text(AIMessage(content="hi")) == "hi"
    assert extract_text(AIMessageChunk(content=[{"type": "text", "text": "x", "index": 0}])) == "x"


def test_extract_usage() -> None:
    msg = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "claude-haiku-4-5"},
    )
    usage = extract_usage(msg, default_model="fallback")
    assert (usage.model, usage.input_tokens, usage.output_tokens) == ("claude-haiku-4-5", 10, 5)


def test_extract_usage_missing_metadata() -> None:
    usage = extract_usage(AIMessage(content="x"), default_model="m")
    assert (usage.model, usage.input_tokens, usage.output_tokens) == ("m", 0, 0)


def test_refusal_raises() -> None:
    with pytest.raises(LLMRefusalError):
        raise_on_refusal(AIMessage(content="", response_metadata={"stop_reason": "refusal"}))
    raise_on_refusal(AIMessage(content="ok", response_metadata={"stop_reason": "end_turn"}))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5", True),
        ("claude-fable-5-1", True),
        ("claude-sonnet-5", False),
        ("claude-haiku-4-5", False),
        ("claude-opus-4-8", False),
    ],
)
def test_server_side_fallback_only_for_supported_models(model: str, expected: bool) -> None:
    assert supports_server_side_fallback(model) is expected
