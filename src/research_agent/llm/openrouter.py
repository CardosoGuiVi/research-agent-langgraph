"""OpenRouter implementation of the LLMProvider port, via langchain-openai's ChatOpenAI.

OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI with a custom `base_url` is enough.
Nothing here touches the network until a method is awaited.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from research_agent.config import Settings
from research_agent.llm.base import (
    LLMError,
    LLMRefusalError,
    Tier,
    Usage,
    extract_text,
    extract_usage,
)


def raise_on_refusal(message: BaseMessage) -> None:
    if message.response_metadata.get("finish_reason") == "content_filter":
        raise LLMRefusalError("model declined the request")


class OpenRouterProvider:
    def __init__(self, settings: Settings) -> None:
        if settings.openrouter_api_key is None:
            raise LLMError("OPENROUTER_API_KEY is not set")
        if not settings.openrouter_model:
            raise LLMError("OPENROUTER_MODEL is not set")
        self._settings = settings
        smart = settings.openrouter_model_smart or settings.openrouter_model
        self._models = {
            Tier.FAST: self._build(settings.openrouter_model),
            Tier.SMART: self._build(smart),
        }

    def _build(self, model: str) -> ChatOpenAI:
        s = self._settings
        return ChatOpenAI(
            model=model,
            api_key=s.openrouter_api_key,
            base_url=s.openrouter_base_url,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
            stream_usage=True,
        )

    def _model(self, tier: Tier, max_tokens: int) -> ChatOpenAI:
        return self._models[tier].model_copy(update={"max_tokens": max_tokens})

    async def structured[T: BaseModel](
        self, schema: type[T], *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> tuple[T, Usage]:
        model = self._model(tier, max_tokens)
        # Native JSON-schema output (`response_format`). Forced tool calls were ignored by
        # DeepSeek in a live eval. The schema goes in as a dict so ChatOpenAI parses the JSON
        # content instead of the SDK's strict `parse` path; pydantic validates it here.
        runnable = model.with_structured_output(
            schema.model_json_schema(), method="json_schema", include_raw=True
        )
        out = cast(
            dict[str, Any],
            await runnable.ainvoke([SystemMessage(system), HumanMessage(user)]),
        )
        raw = cast(AIMessage, out["raw"])
        raise_on_refusal(raw)
        usage = extract_usage(raw, default_model=model.model_name)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise LLMError(f"structured output did not match {schema.__name__}")
        try:
            return schema.model_validate(out["parsed"]), usage
        except ValidationError as exc:
            raise LLMError(f"structured output did not match {schema.__name__}") from exc

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
        tool_choice = None if allow_tool_calls else "none"
        runnable: Any = model.bind_tools(list(tools), tool_choice=tool_choice) if tools else model
        msg = cast(AIMessage, await runnable.ainvoke([SystemMessage(system), *messages]))
        raise_on_refusal(msg)
        return msg, extract_usage(msg, default_model=model.model_name)

    async def stream_text(
        self, *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> AsyncIterator[str | Usage]:
        model = self._model(tier, max_tokens)
        final: AIMessage | None = None
        served = model.model_name
        async for chunk in model.astream([SystemMessage(system), HumanMessage(user)]):
            final = chunk if final is None else cast(AIMessage, final + chunk)
            # Every chunk repeats model_name, so the merged message has it concatenated.
            served = str(chunk.response_metadata.get("model_name") or served)
            if text := extract_text(chunk):
                yield text
        if final is not None:
            raise_on_refusal(final)
            yield replace(extract_usage(final, default_model=served), model=served)
