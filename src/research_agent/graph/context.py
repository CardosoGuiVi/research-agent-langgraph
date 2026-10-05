"""Dependencies (process-wide) and the per-run scope (budget, cache, metrics)."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from langgraph.config import get_config
from langgraph.runtime import Runtime

from research_agent.config import Settings
from research_agent.llm.base import LLMProvider, Usage
from research_agent.llm.pricing import estimate_cost_usd
from research_agent.logging import get_logger
from research_agent.tools.budget import BudgetedSearch, RunBudget
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.search import SearchProvider

if TYPE_CHECKING:
    from research_agent.graph.state import ResearchState

_log = get_logger(__name__)


@dataclass
class LLMCall:
    node: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class RunMetrics:
    started: float = field(default_factory=time.perf_counter)
    nodes: list[str] = field(default_factory=list)
    tool_calls: Counter[str] = field(default_factory=Counter)
    llm_calls: list[LLMCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, int]] = {}
        cost: float | None = 0.0
        for c in self.llm_calls:
            m = by_model.setdefault(c.model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            m["input_tokens"] += c.input_tokens
            m["output_tokens"] += c.output_tokens
            m["calls"] += 1
        for model, m in by_model.items():
            model_cost = estimate_cost_usd(model, m["input_tokens"], m["output_tokens"])
            cost = None if (cost is None or model_cost is None) else cost + model_cost
        return {
            "nodes_executed": list(self.nodes),
            "tool_calls": dict(self.tool_calls),
            "llm_calls": len(self.llm_calls),
            "input_tokens": sum(c.input_tokens for c in self.llm_calls),
            "output_tokens": sum(c.output_tokens for c in self.llm_calls),
            "tokens_by_model": by_model,
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
            "latency_s": round(time.perf_counter() - self.started, 3),
            "errors": list(self.errors),
        }


@dataclass
class RunScope:
    """Run-scoped state that must not be checkpointed: limits, search cache, metrics."""

    run_id: str
    budget: RunBudget
    search: BudgetedSearch
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def record_llm(self, node: str, usage: Usage) -> None:
        self.budget.add_tokens(usage.total_tokens)
        self.metrics.llm_calls.append(
            LLMCall(node, usage.model, usage.input_tokens, usage.output_tokens)
        )


@dataclass
class GraphDeps:
    settings: Settings
    llm: LLMProvider
    search_provider: SearchProvider
    fetcher: PageFetcher
    _fallback_scopes: dict[str, RunScope] = field(default_factory=dict)

    def new_scope(self, run_id: str | None = None) -> RunScope:
        s = self.settings
        budget = RunBudget(
            max_searches=s.max_searches_per_run,
            max_fetches=s.max_fetches_per_run,
            max_tokens=s.max_tokens_per_run,
        )
        return RunScope(
            run_id=run_id or uuid.uuid4().hex,
            budget=budget,
            search=BudgetedSearch(self.search_provider, budget),
        )

    def scope(self, runtime: Runtime[RunScope], *, reset: bool = False) -> RunScope:
        """The run's scope. The API always passes one as `context`.

        Tools that invoke the graph without a context (e.g. LangGraph Studio) get a
        per-thread fallback scope, reset by the planner at the start of each run.
        """
        if runtime.context is not None:
            return runtime.context
        thread_id = str(get_config().get("configurable", {}).get("thread_id", "default"))
        if reset or thread_id not in self._fallback_scopes:
            if len(self._fallback_scopes) > 64:
                self._fallback_scopes.clear()
            self._fallback_scopes[thread_id] = self.new_scope()
        return self._fallback_scopes[thread_id]


Emit = Callable[[dict[str, Any]], None]


class StateNode(Protocol):
    """A node taking the shared graph state (matches LangGraph's runtime-injection signature)."""

    async def __call__(
        self, state: ResearchState, *, runtime: Runtime[RunScope]
    ) -> dict[str, Any]: ...


def emitter(runtime: Runtime[RunScope]) -> Emit:
    """Custom stream events (stream_mode="custom"); no-op when nobody is streaming."""
    writer = runtime.stream_writer

    def emit(event: dict[str, Any]) -> None:
        try:
            writer(event)
        except Exception:  # pragma: no cover - streaming must never break a run
            _log.debug("stream_writer_failed", exc_info=True)

    return emit
