import pytest

from research_agent.config import LLMProviderName, SearchProviderName, Settings


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


def test_llm_provider_defaults_to_anthropic_and_needs_no_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = _settings()
    assert s.llm_provider is LLMProviderName.ANTHROPIC
    assert s.openrouter_api_key is None
    assert s.openrouter_model is None
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_openrouter_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-very-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model")
    s = _settings()
    assert s.llm_provider is LLMProviderName.OPENROUTER
    assert s.openrouter_model == "vendor/model"
    assert "or-very-secret" not in repr(s)


def test_unknown_llm_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="llm_provider"):
        _settings(llm_provider="nope")
