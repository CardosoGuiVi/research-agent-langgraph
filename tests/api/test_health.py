from httpx import ASGITransport, AsyncClient

from research_agent.api.app import create_app
from tests.fakes import build_test_container


async def test_health_ok_and_request_id_echoed() -> None:
    app = create_app(container=build_test_container())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/health", headers={"X-Request-ID": "req-123"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["search_provider"] in {"tavily", "duckduckgo"}
    assert r.headers["X-Request-ID"] == "req-123"


async def test_request_id_generated_when_missing() -> None:
    app = create_app(container=build_test_container())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/health")
    assert r.headers["X-Request-ID"]
