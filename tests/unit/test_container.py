from pathlib import Path

import pytest

from research_agent.api.app import create_app
from research_agent.container import build_container, make_llm, make_search_provider
from research_agent.llm.anthropic import AnthropicLLM
from research_agent.llm.base import LLMError
from research_agent.llm.openrouter import OpenRouterProvider
from research_agent.tools.search import DuckDuckGoSearch, TavilySearch
from tests.fakes import make_settings


def test_search_provider_selection() -> None:
    assert isinstance(make_search_provider(make_settings()), DuckDuckGoSearch)
    assert isinstance(make_search_provider(make_settings(tavily_api_key="tvly-x")), TavilySearch)


async def test_build_container_with_real_adapters_and_lifespan_closes_clients() -> None:
    container = build_container(make_settings())
    assert isinstance(container.service.deps.llm, AnthropicLLM)
    app = create_app(container=container)
    async with app.router.lifespan_context(app):
        pass
    assert all(c.is_closed for c in container._http_clients)


def test_llm_provider_defaults_to_anthropic() -> None:
    assert isinstance(make_llm(make_settings()), AnthropicLLM)


def test_llm_provider_openrouter_is_selected_by_setting() -> None:
    settings = make_settings(
        llm_provider="openrouter",
        anthropic_api_key=None,
        openrouter_api_key="or-test",
        openrouter_model="vendor/model",
    )
    assert isinstance(make_llm(settings), OpenRouterProvider)
    container = build_container(settings)
    assert isinstance(container.service.deps.llm, OpenRouterProvider)


def test_openrouter_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_llm(make_settings(llm_provider="openrouter", openrouter_model="vendor/model"))


def test_graph_depends_only_on_the_llm_provider_port() -> None:
    """Nodes and the service import `llm.base`, never a concrete adapter (wired in container)."""
    src = Path(__file__).parents[2] / "src" / "research_agent"
    offenders = [
        str(path.relative_to(src))
        for path in [*src.joinpath("graph").rglob("*.py"), src / "service.py"]
        if "research_agent.llm.anthropic" in path.read_text()
        or "research_agent.llm.openrouter" in path.read_text()
    ]
    assert offenders == []
