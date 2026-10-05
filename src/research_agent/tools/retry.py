"""Timeout + retry-with-exponential-backoff helper for tool calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from research_agent.logging import get_logger
from research_agent.tools.errors import ToolError, TransientError

__all__ = ["TransientError", "call_with_retries"]

_log = get_logger(__name__)


async def call_with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    timeout_s: float,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    timeout_error: type[ToolError] = TransientError,
) -> T:
    """Run `fn` with a per-attempt timeout, retrying only transient errors and timeouts.

    When every attempt times out, raises `timeout_error` (a ToolError) so callers that handle
    tool failures also handle timeouts.
    """
    retrying = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=base_delay_s, max=max_delay_s, jitter=base_delay_s),
        retry=retry_if_exception_type((TransientError, TimeoutError)),
        reraise=True,
        before_sleep=lambda rs: _log.warning(
            "tool_retry",
            attempt=rs.attempt_number,
            error=type(rs.outcome.exception()).__name__ if rs.outcome else None,
        ),
    )
    try:
        async for attempt in retrying:
            with attempt:
                async with asyncio.timeout(timeout_s):
                    return await fn()
    except TimeoutError as exc:
        raise timeout_error(f"timed out after {attempts} attempt(s) of {timeout_s}s") from exc
    raise AssertionError("unreachable")  # pragma: no cover
