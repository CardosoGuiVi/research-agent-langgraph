import pytest

from research_agent.llm.pricing import estimate_cost_usd


def test_known_model_cost() -> None:
    # Haiku 4.5: $1 / $5 per MTok
    assert estimate_cost_usd("claude-haiku-4-5", 1_000_000, 200_000) == pytest.approx(2.0)


def test_opus_5_cost() -> None:
    assert estimate_cost_usd("claude-opus-5", 10_000, 2_000) == pytest.approx(0.1)


def test_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("some-future-model", 10, 10) is None


def test_dated_model_ids_resolve_to_base_price() -> None:
    assert estimate_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)
    # "claude-opus-5-5" must not be priced as "claude-opus-5"
    assert estimate_cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
