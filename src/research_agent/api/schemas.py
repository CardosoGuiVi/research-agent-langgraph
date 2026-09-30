"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    search_provider: str
