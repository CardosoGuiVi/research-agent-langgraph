from httpx import AsyncClient

from research_agent.container import Container
from tests.api.conftest import fake_llm, parse_sse


async def test_post_research_returns_report_and_usage(client: AsyncClient) -> None:
    r = await client.post("/research", json={"question": "Observability for LLM apps?"})
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"]
    assert body["run_id"]
    report = body["report"]
    assert report["question"] == "Observability for LLM apps?"
    assert report["summary"]
    assert report["sources"][0]["url"].startswith("https://")
    usage = body["usage"]
    assert usage["llm_calls"] > 0
    assert usage["input_tokens"] > 0
    assert usage["estimated_cost_usd"] > 0
    assert usage["nodes_executed"][0] == "planner"
    assert usage["nodes_executed"][-1] == "answer"
    assert usage["tool_calls"]["web_search"] >= 2


async def test_question_validation(client: AsyncClient) -> None:
    assert (await client.post("/research", json={"question": "  "})).status_code == 422
    assert (await client.post("/research", json={"question": "x" * 2001})).status_code == 422
    bad_thread = {"question": "valid question", "thread_id": "../etc"}
    assert (await client.post("/research", json=bad_thread)).status_code == 422


async def test_follow_up_on_same_thread_and_thread_view(client: AsyncClient) -> None:
    first = (await client.post("/research", json={"question": "First question?"})).json()
    thread_id = first["thread_id"]
    second = await client.post(
        "/research", json={"question": "Follow-up question?", "thread_id": thread_id}
    )
    assert second.status_code == 200
    assert second.json()["run_id"] != first["run_id"]

    r = await client.get(f"/threads/{thread_id}")
    assert r.status_code == 200
    view = r.json()
    assert view["thread_id"] == thread_id
    assert view["questions"] == ["First question?", "Follow-up question?"]
    assert view["report"]["question"] == "Follow-up question?"
    assert len(view["notes"]) >= 4
    assert view["sources"]
    assert view["history"]
    assert {"checkpoint_id", "step", "next", "created_at"} <= set(view["history"][0])


async def test_unknown_thread_is_404(client: AsyncClient) -> None:
    assert (await client.get("/threads/does-not-exist")).status_code == 404


async def test_stream_emits_progress_tokens_and_report(client: AsyncClient) -> None:
    async with client.stream(
        "POST", "/research/stream", json={"question": "Observability?", "thread_id": "s1"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = (await r.aread()).decode()
    events = parse_sse(body)
    names = [n for n, _ in events]
    assert names[0] == "run_started"
    assert events[0][1]["thread_id"] == "s1"
    assert {"node", "search", "fetch", "token"} <= set(names)
    assert names[-2:] == ["report", "done"]
    tokens = "".join(d["text"] for n, d in events if n == "token")
    assert "Answer" in tokens
    report_event = events[-2][1]
    assert report_event["report"]["summary"]
    assert report_event["usage"]["llm_calls"] > 0
    node_events = [d for n, d in events if n == "node" and d["status"] == "finished"]
    assert node_events[0]["node"] == "planner"


async def test_concurrent_run_on_same_thread_is_rejected(
    container: Container, client: AsyncClient
) -> None:
    lock = container.service.thread_lock("busy")
    await lock.acquire()
    try:
        r = await client.post("/research", json={"question": "Is it busy?", "thread_id": "busy"})
        assert r.status_code == 409
        r = await client.post(
            "/research/stream", json={"question": "Is it busy?", "thread_id": "busy"}
        )
        assert r.status_code == 409
    finally:
        lock.release()


async def test_unexpected_error_returns_500_without_leaking_details() -> None:
    from httpx import ASGITransport

    from research_agent.api.app import create_app
    from tests.fakes import build_test_container

    container = build_test_container(llm=fake_llm())

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail sk-ant-xyz")

    container.service.graph.ainvoke = boom  # type: ignore[method-assign]
    app = create_app(container=container)
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t"
    ) as c:
        r = await c.post("/research", json={"question": "Will fail?"})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert r.json()["request_id"]


async def test_stream_error_event_on_unexpected_failure() -> None:
    from httpx import ASGITransport

    from research_agent.api.app import create_app
    from tests.fakes import build_test_container

    container = build_test_container(llm=fake_llm())

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    container.service.graph.astream = boom  # type: ignore[method-assign]
    app = create_app(container=container)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/research/stream", json={"question": "Will fail?"})
    events = parse_sse(r.text)
    assert events[-1][0] == "error"
    assert "secret" not in r.text


async def test_run_released_lock_after_completion(
    client: AsyncClient, container: Container
) -> None:
    await client.post("/research", json={"question": "Is it busy?", "thread_id": "t-lock"})
    assert not container.service.thread_lock("t-lock").locked()
    await client.post("/research/stream", json={"question": "Is it busy?", "thread_id": "t-lock2"})
    assert not container.service.thread_lock("t-lock2").locked()
