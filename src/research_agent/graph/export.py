"""Export the compiled graph as Mermaid: `python -m research_agent.graph.export docs/graph.mmd`.

Builds the graph with inert adapters (nothing is called), so no API keys are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from research_agent.config import Settings
from research_agent.graph.builder import build_graph
from research_agent.graph.context import GraphDeps
from research_agent.llm.base import LLMProvider
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.search import SearchProvider


def mermaid() -> str:
    deps = GraphDeps(
        settings=Settings(_env_file=None),
        llm=cast(LLMProvider, None),
        search_provider=cast(SearchProvider, None),
        fetcher=cast(PageFetcher, None),
    )
    return build_graph(deps).get_graph().draw_mermaid()


def main(argv: list[str]) -> None:
    text = mermaid()
    if len(argv) > 1:
        Path(argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[1]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv)
