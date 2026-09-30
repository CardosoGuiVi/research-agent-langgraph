from httpx import AsyncClient


async def test_index_serves_the_single_page_ui(client: AsyncClient) -> None:
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<form id="ask"' in r.text
    js = await client.get("/static/app.js")
    assert js.status_code == 200
    assert "/research/stream" in js.text
    assert (await client.get("/static/styles.css")).status_code == 200
