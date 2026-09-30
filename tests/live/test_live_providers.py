"""Real API calls. Skipped by default; run with `make test-live` (costs a few cents)."""

import httpx
import pytest
from ddgs import DDGS
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from research_agent.config import Settings
from research_agent.llm.anthropic import AnthropicLLM
from research_agent.llm.base import Tier, Usage
from research_agent.tools.fetch import HttpPageFetcher
from research_agent.tools.search import DuckDuckGoSearch

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = Settings()
    if s.anthropic_api_key is None:
        pytest.skip("ANTHROPIC_API_KEY not configured")
    return s


class Capital(BaseModel):
    city: str


async def test_structured_output_fast_model(settings: Settings) -> None:
    out, usage = await AnthropicLLM(settings).structured(
        Capital,
        system="Answer briefly.",
        user="Capital of France?",
        tier=Tier.FAST,
        max_tokens=100,
    )
    assert "paris" in out.city.lower()
    assert usage.input_tokens > 0


async def test_tool_step_calls_the_tool(settings: Settings) -> None:
    tool = {
        "name": "web_search",
        "description": "Search the web.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    ai, _ = await AnthropicLLM(settings).tool_step(
        [HumanMessage("Search the web for 'OpenTelemetry GenAI semantic conventions'.")],
        system="Use tools when asked.",
        tools=[tool],
        tier=Tier.FAST,
        max_tokens=200,
    )
    assert ai.tool_calls
    assert ai.tool_calls[0]["name"] == "web_search"


async def test_stream_text_smart_model(settings: Settings) -> None:
    items = [
        i
        async for i in AnthropicLLM(settings).stream_text(
            system="Be terse.", user="Say hello in one word.", tier=Tier.SMART, max_tokens=2000
        )
    ]
    text = "".join(i for i in items if isinstance(i, str))
    assert text.strip()
    assert isinstance(items[-1], Usage)
    assert items[-1].output_tokens > 0


async def test_duckduckgo_search() -> None:
    results = await DuckDuckGoSearch(DDGS(), timeout_s=15, attempts=2).search(
        "LangGraph documentation", max_results=3
    )
    assert results


async def test_fetch_real_page() -> None:
    async with httpx.AsyncClient() as client:
        page = await HttpPageFetcher(
            client, max_bytes=2_000_000, max_chars=4_000, timeout_s=15, attempts=2
        ).fetch("https://www.python.org/")
    assert "Python" in page.text
