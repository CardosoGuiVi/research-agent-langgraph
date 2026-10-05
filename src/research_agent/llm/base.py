"""The LLMProvider port used by graph nodes. Nodes never touch a vendor SDK directly.

Keeping this interface narrow (three operations the graph actually needs) makes nodes easy to
test with a scripted fake and keeps provider details in one adapter (Anthropic, OpenRouter).
The adapter is chosen by `LLM_PROVIDER` in `container.make_llm`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel


class Tier(StrEnum):
    FAST = "fast"  # planning, tool routing, coverage analysis, judging
    SMART = "smart"  # final synthesis


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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


class LLMError(Exception):
    pass


class LLMRefusalError(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


class LLMProvider(Protocol):
    async def structured[T: BaseModel](
        self, schema: type[T], *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> tuple[T, Usage]:
        """One call whose output is validated against `schema`."""
        ...

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
        """One turn of a tool-calling loop; the caller executes any `tool_calls`.

        `tools` are Anthropic-format tool definitions (name, description, input_schema);
        LangChain adapters convert them to the provider's format.
        `allow_tool_calls=False` keeps the tools defined (required when the history has tool
        blocks) but sends `tool_choice: none` so the model must answer in text.
        """
        ...

    def stream_text(
        self, *, system: str, user: str, tier: Tier, max_tokens: int
    ) -> AsyncIterator[str | Usage]:
        """Yield text deltas, then a final `Usage`."""
        ...
