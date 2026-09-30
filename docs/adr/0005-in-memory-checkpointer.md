# 5. In-memory checkpointer for thread memory (for now)

- Status: accepted
- Date: 2026-09-30

## Context and problem statement

Follow-up questions on a thread must reuse earlier findings, and `GET /threads/{id}` must show
state and history. LangGraph checkpointers provide both, keyed by `thread_id`.

## Decision outcome

Use `InMemorySaver` with an explicit msgpack allowlist of our domain types (only these can be
deserialized). Runs on one thread are serialized in the service (HTTP 409 when busy).

This is the smallest thing that validates the feature in a single container. Threads are lost on
restart and memory is per process.

**Trigger to revisit:** more than one API replica, or users who expect threads to survive
restarts. Then switch to `langgraph-checkpoint-postgres` (one-line change in
`graph/builder.py::new_checkpointer`) plus a Postgres service in compose, and move the
per-thread lock to the database (advisory lock).
