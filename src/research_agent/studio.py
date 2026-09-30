"""Entry point for LangGraph Studio (`langgraph dev`, see langgraph.json).

Studio supplies its own checkpointer and does not pass our run `context`, so nodes fall back to a
per-thread RunScope (budgets still apply per run; see GraphDeps.scope).
"""

from research_agent.config import get_settings
from research_agent.container import build_deps
from research_agent.graph.builder import build_graph

graph = build_graph(build_deps(get_settings()))
