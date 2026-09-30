"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from research_agent import __version__
from research_agent.api.middleware import REQUEST_ID_HEADER, RequestIdMiddleware
from research_agent.api.routes import router
from research_agent.config import get_settings
from research_agent.container import Container, build_container
from research_agent.logging import configure_logging, get_logger

_log = get_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(container: Container | None = None) -> FastAPI:
    if container is None:
        settings = get_settings()
        configure_logging(settings.log_level, json=settings.log_json)
        container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _log.info(
            "app_started",
            version=__version__,
            search_provider=container.settings.search_provider.value,
        )
        yield
        await container.aclose()

    app = FastAPI(title="Research Agent", version=__version__, lifespan=lifespan)
    app.state.container = container
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the details server-side; never echo them (they may contain provider errors).
        request_id = getattr(request.state, "request_id", None)
        _log.error(
            "unhandled_error", request_id=request_id, error_type=type(exc).__name__, exc_info=exc
        )
        return JSONResponse(
            {"detail": "internal error", "request_id": request_id},
            status_code=500,
            headers={REQUEST_ID_HEADER: request_id} if request_id else None,
        )

    return app
