"""Request-ID middleware: accepts or generates an ID and binds it to every log line."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from research_agent.logging import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
_log = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        # Accept only short, safe IDs from clients; otherwise mint our own.
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.isprintable() else ""
        request_id = request_id or uuid.uuid4().hex
        request.state.request_id = request_id  # for the 500 handler, which runs outside us
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        if request.url.path != "/health":
            _log.info(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        return response
