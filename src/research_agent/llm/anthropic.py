"""Anthropic implementation of the LLM port, via langchain-anthropic's ChatAnthropic."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from research_agent.config import Settings
from research_agent.llm.base import LLMError, LLMRefusalError, Tier, Usage

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallbacks are available on the Opus 5 / Fable families.
_FALLBACK_MODEL_PREFIXES = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")


def supports_server_side_fallback(model: str) -> bool:
    return model.startswith(_FALLBACK_MODEL_PREFIXES)


def extract_text(message: BaseMessage) -> str:
    """Concatenate text blocks only (drops thinking / tool_use blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def extract_usage(message: AIMessage, *, default_model: str) -> Usage:
    meta = message.usage_metadata
    model = str(message.response_metadata.get("model_name") or default_model)
    return Usage(
        model=model,
        input_tokens=meta["input_tokens"] if meta else 0,
        output_tokens=meta["output_tokens"] if meta else 0,
    )


def raise_on_refusal(message: BaseMessage) -> None:
    if message.response_metadata.get("stop_reason") == "refusal":
        raise LLMRefusalError("model declined the request")


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        if settings.anthropic_api_key is None:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._settings = settings
        self._models = {
            Tier.FAST: self._build(settings.llm_model_fast, effort=None),
            Tier.SMART: self._build(settings.llm_model_smart, effort=settings.llm_smart_effort),
        }

    def _build(self, model: str, *, effort: str | None) -> ChatAnthropic:
        s = self._settings
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": s.anthropic_api_key,
            "timeout": s.llm_timeout_s,
            "max_retries": s.llm_max_retries,  # SDK retries 408/409/429/5xx with backoff
            "stream_usage": True,
        }
        if effort is not None:
            kwargs["effort"] = effort
        if supports_server_side_fallback(model):
            # If a safety classifier declines, the API retries on a substitute model.
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["model_kwargs"] = {"fallbacks": "default"}
        return ChatAnthropic(**kwargs)

    def _model(self, tier: Tier, max_tokens: int) -> ChatAnthropic:
        return self._models[tier].model_copy(update={"max_tokens": max_tokens})

    async def structured[T: BaseModel](
        self, schema: type[T], *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> tuple[T, Usage]:
        model = self._model(tier, max_tokens)
        runnable = model.with_structured_output(schema, method="json_schema", include_raw=True)
        out = cast(
            dict[str, Any],
            await runnable.ainvoke([SystemMessage(system), HumanMessage(user)]),
        )
        raw = cast(AIMessage, out["raw"])
        raise_on_refusal(raw)
        usage = extract_usage(raw, default_model=model.model)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise LLMError(f"structured output did not match {schema.__name__}")
        return cast(T, out["parsed"]), usage

    async def tool_step(
        self,
        messages: Sequence[BaseMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
        tier: Tier,
        max_tokens: int,
        allow_tool_calls: bool = True,
    ) -> tuple[AIMessage, Usage]:
        model = self._model(tier, max_tokens)
        tool_choice = None if allow_tool_calls else {"type": "none"}
        runnable: Any = model.bind_tools(list(tools), tool_choice=tool_choice) if tools else model
        msg = cast(AIMessage, await runnable.ainvoke([SystemMessage(system), *messages]))
        raise_on_refusal(msg)
        return msg, extract_usage(msg, default_model=model.model)

    async def stream_text(
        self, *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> AsyncIterator[str | Usage]:
        model = self._model(tier, max_tokens)
        final: AIMessage | None = None
        async for chunk in model.astream([SystemMessage(system), HumanMessage(user)]):
            final = chunk if final is None else cast(AIMessage, final + chunk)
            if text := extract_text(chunk):
                yield text
        if final is not None:
            raise_on_refusal(final)
            yield extract_usage(final, default_model=model.model)
