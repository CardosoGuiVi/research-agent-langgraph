import asyncio

import pytest

from research_agent.tools.errors import TransientSearchError
from research_agent.tools.retry import TransientError, call_with_retries


async def test_retries_transient_errors_then_succeeds() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientError("503")
        return "ok"

    assert await call_with_retries(flaky, attempts=3, timeout_s=1, base_delay_s=0) == "ok"
    assert calls == 3


async def test_gives_up_after_max_attempts() -> None:
    async def always_down() -> str:
        raise TransientError("down")

    with pytest.raises(TransientError):
        await call_with_retries(always_down, attempts=2, timeout_s=1, base_delay_s=0)


async def test_non_transient_errors_are_not_retried() -> None:
    calls = 0

    async def bad_request() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("400")

    with pytest.raises(ValueError):
        await call_with_retries(bad_request, attempts=5, timeout_s=1, base_delay_s=0)
    assert calls == 1


async def test_timeouts_are_retried_as_transient() -> None:
    calls = 0

    async def slow_then_fast() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return "ok"

    assert (
        await call_with_retries(slow_then_fast, attempts=2, timeout_s=0.01, base_delay_s=0) == "ok"
    )


async def test_exhausted_timeouts_become_a_tool_error() -> None:
    """A raw TimeoutError is not a ToolError, so it used to abort the whole research branch."""

    async def always_slow() -> str:
        await asyncio.sleep(1)
        return "never"

    with pytest.raises(TransientError, match="timed out"):
        await call_with_retries(always_slow, attempts=2, timeout_s=0.01, base_delay_s=0)


async def test_timeout_error_type_is_chosen_by_the_caller() -> None:
    async def always_slow() -> str:
        await asyncio.sleep(1)
        return "never"

    with pytest.raises(TransientSearchError):
        await call_with_retries(
            always_slow,
            attempts=1,
            timeout_s=0.01,
            base_delay_s=0,
            timeout_error=TransientSearchError,
        )
