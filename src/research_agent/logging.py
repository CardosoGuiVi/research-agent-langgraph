"""Structured JSON logging (structlog) with secret redaction and request/run correlation IDs."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_KEY = re.compile(r"(authorization|api[-_]?key|secret|token_value|password|cookie)", re.I)
# Anthropic (sk-ant-...), Tavily (tvly-...) and generic bearer tokens.
_SENSITIVE_VALUE = re.compile(r"(sk-ant-[A-Za-z0-9_\-]{6,}|tvly-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)")
_MASK = "***"


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub(_MASK, value)
    if isinstance(value, dict):
        return {
            k: (_MASK if isinstance(k, str) and _SENSITIVE_KEY.search(k) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_redact(v) for v in value)
    return value


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: mask sensitive keys and key-like substrings anywhere in the event."""
    for key in list(event_dict):
        if _SENSITIVE_KEY.search(key):
            event_dict[key] = _MASK
        else:
            event_dict[key] = _redact(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # request_id / run_id / thread_id
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    # Silence chatty third-party loggers that could echo request metadata.
    for name in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
