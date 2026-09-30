"""Versioned prompt files.

Each `<name>.md` has YAML front matter (`name`, `version`) and two sections introduced by
`<!-- system -->` and `<!-- user -->`. User templates use `string.Template` ($var) syntax so
JSON and markdown braces need no escaping. Bump `version` whenever the text changes; the
version is logged with each run so eval results can be tied to a prompt revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from string import Template
from typing import Any

import yaml

PROMPT_NAMES = ("planner", "research", "analysis", "answer", "judge")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    user_template: str

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def render_user(self, **values: Any) -> str:
        # substitute() raises KeyError on missing variables: fail loudly, never send "$var".
        return Template(self.user_template).substitute(**values)


@cache
def load_prompt(name: str) -> Prompt:
    raw = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    _, front, body = raw.split("---", 2)
    meta = yaml.safe_load(front)
    system_part, user_part = body.split("<!-- user -->", 1)
    system = system_part.replace("<!-- system -->", "", 1).strip()
    return Prompt(
        name=str(meta["name"]),
        version=int(meta["version"]),
        system=system,
        user_template=user_part.strip(),
    )
