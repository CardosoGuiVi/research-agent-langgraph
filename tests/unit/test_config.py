import pytest

from research_agent.config import SearchProviderName, Settings


def _settings(**overrides: object) -> Settings:
    # _env_file=None keeps tests independent of any local .env
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_safe_and_cheap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    s = _settings(anthropic_api_key="sk-test")
    assert s.max_iterations >= 1
    assert s.max_searches_per_run > 0
    assert s.llm_model_fast
    assert s.llm_model_smart


def test_duckduckgo_is_used_when_tavily_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert _settings().search_provider is SearchProviderName.DUCKDUCKGO


def test_tavily_is_used_when_key_present() -> None:
    s = _settings(tavily_api_key="tvly-test")
    assert s.search_provider is SearchProviderName.TAVILY


def test_secrets_are_not_leaked_in_repr() -> None:
    s = _settings(anthropic_api_key="sk-very-secret", tavily_api_key="tvly-very-secret")
    assert "sk-very-secret" not in repr(s)
    assert "tvly-very-secret" not in repr(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-very-secret"


def test_empty_tavily_key_counts_as_absent() -> None:
    assert _settings(tavily_api_key="").search_provider is SearchProviderName.DUCKDUCKGO
