"""Per-model token prices for cost estimation (USD per 1M tokens).

Source: Anthropic pricing page, checked 2026-09-30. Update when models/prices change.
Estimates ignore prompt-cache discounts, so they are an upper bound.
"""

from __future__ import annotations

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model id: (input, output)
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_KEYS_LONGEST_FIRST = sorted(PRICES_PER_MTOK, key=len, reverse=True)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    # Longest-prefix match so dated IDs (e.g. "claude-haiku-4-5-20251001") resolve too.
    key = next((k for k in _KEYS_LONGEST_FIRST if model == k or model.startswith(k + "-")), None)
    if key is None:
        return None
    prices = PRICES_PER_MTOK[key]
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
