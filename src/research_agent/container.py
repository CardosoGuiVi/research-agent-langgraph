"""Composition root: wires settings and adapters into the objects the API uses."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.config import Settings


@dataclass(frozen=True)
class Container:
    settings: Settings


def build_container(settings: Settings) -> Container:
    return Container(settings=settings)
