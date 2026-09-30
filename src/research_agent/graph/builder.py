"""Graph assembly: START -> planner -> research (parallel) -> analysis -> (research | answer)."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent import domain
from research_agent.graph.context import GraphDeps, RunScope
from research_agent.graph.nodes.analysis import make_analysis
from research_agent.graph.nodes.answer import make_answer
from research_agent.graph.nodes.planner import make_planner
from research_agent.graph.nodes.research import make_research
from research_agent.graph.routing import route_after_analysis, route_after_planner
from research_agent.graph.state import ResearchInput, ResearchState, ResearchTask

ResearchGraph = CompiledStateGraph[ResearchState, RunScope, ResearchInput, ResearchState]

# Explicit allowlist: the checkpointer only deserializes our own domain types.
_CHECKPOINT_TYPES: list[type[Any]] = [
    domain.ResearchPlan,
    domain.SubQuestion,
    domain.ResearchNote,
    domain.Source,
    domain.SubQuestionAssessment,
    domain.Report,
    domain.ReportSection,
    domain.SourceRef,
]


def new_checkpointer() -> BaseCheckpointSaver[str]:
    """In-memory checkpointer (see ADR 0002 for why not Postgres yet)."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES))


def build_graph(
    deps: GraphDeps, checkpointer: BaseCheckpointSaver[str] | None = None
) -> ResearchGraph:
    graph = StateGraph(ResearchState, context_schema=RunScope, input_schema=ResearchInput)
    graph.add_node("planner", make_planner(deps))
    graph.add_node("research", make_research(deps), input_schema=ResearchTask)
    graph.add_node("analysis", make_analysis(deps))
    graph.add_node("answer", make_answer(deps))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner, ["research", "answer"])
    graph.add_edge("research", "analysis")  # fan-in: runs once all parallel branches finish
    graph.add_conditional_edges(
        "analysis",
        partial(route_after_analysis, max_iterations=deps.settings.max_iterations),
        ["research", "answer"],
    )
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer, name="research-agent")
