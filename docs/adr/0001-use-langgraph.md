# 1. Use LangGraph for agent orchestration

- Status: accepted
- Date: 2026-09-30

## Context and problem statement

The agent is a loop with branching: plan, research several sub-questions, judge coverage, and
either research again or answer. We need explicit control flow, parallel branches, a hard cap on
loops, per-thread memory for follow-ups, streaming of intermediate progress, and something we can
test node by node.

## Considered options

1. Hand-written asyncio orchestration around the Anthropic SDK.
2. A single tool-calling agent loop (the model decides everything).
3. LangGraph `StateGraph` with `langchain-anthropic`.

## Decision outcome

Chosen option: **LangGraph**, because it gives us, without custom plumbing:

- Explicit graph topology with typed state and reducers (the `sources` reducer assigns citation
  ids deterministically at fan-in).
- `Send` for dynamic fan-out (one branch per sub-question) and automatic fan-in.
- Conditional edges as plain functions, so routing and the iteration cap are unit tested
  without any LLM.
- Checkpointers keyed by `thread_id` for follow-up memory, plus state history for `GET /threads`.
- Runtime `context` for run-scoped dependencies (budgets, cache, metrics) and `stream_mode="custom"`
  for progress events.
- LangGraph Studio for visual debugging.

### Consequences

- Good: the control flow the reviewer sees in the Mermaid diagram is the control flow that runs.
- Good: nodes are small async functions behind our own `LLM` port, so they run in tests with a
  scripted fake.
- Bad: another fast-moving dependency. Mitigated by pinning versions and keeping LangGraph usage
  to core primitives (StateGraph, Send, Runtime, checkpointer).
- Bad: langchain-anthropic sits between us and the Anthropic SDK. We confine it to one adapter
  (`llm/anthropic.py`) so it could be replaced by the SDK directly.

Option 2 was rejected because a free-form agent gives no hard guarantees on iteration count,
parallelism or cost; option 1 would re-implement checkpointing, streaming and fan-in.
