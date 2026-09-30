"""HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from research_agent import __version__
from research_agent.api.schemas import HealthResponse
from research_agent.container import Container

router = APIRouter()


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        search_provider=container.settings.search_provider.value,
    )
