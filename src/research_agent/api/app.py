"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from research_agent import __version__
from research_agent.api.middleware import RequestIdMiddleware
from research_agent.api.routes import router
from research_agent.config import get_settings
from research_agent.container import Container, build_container
from research_agent.logging import configure_logging


def create_app(container: Container | None = None) -> FastAPI:
    if container is None:
        settings = get_settings()
        configure_logging(settings.log_level, json=settings.log_json)
        container = build_container(settings)
    app = FastAPI(title="Research Agent", version=__version__)
    app.state.container = container
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    return app
