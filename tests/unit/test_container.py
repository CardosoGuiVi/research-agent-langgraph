from research_agent.api.app import create_app
from research_agent.container import build_container, make_search_provider
from research_agent.llm.anthropic import AnthropicLLM
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
