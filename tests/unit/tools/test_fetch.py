import httpx
import pytest
import respx

from research_agent.tools.errors import FetchError, TransientFetchError
from research_agent.tools.fetch import HttpPageFetcher

ARTICLE = """
<html><head><title>Observability for LLM apps</title></head>
<body><nav>Home | About</nav>
<article><h1>Observability for LLM apps</h1>
<p>Tracing every model call with OpenTelemetry lets teams see latency and token usage.</p>
<p>Evaluation pipelines catch regressions before they reach production users.</p>
</article><footer>Copyright</footer></body></html>
"""


async def public_resolver(host: str) -> list[str]:
    return {"internal.example": ["10.0.0.5"], "meta.example": ["169.254.169.254"]}.get(
        host, ["93.184.216.34"]
    )


def make_fetcher(client: httpx.AsyncClient, **kw: object) -> HttpPageFetcher:
    params: dict[str, object] = {
        "resolver": public_resolver,
        "max_bytes": 100_000,
        "max_chars": 10_000,
        "timeout_s": 2,
        "attempts": 2,
        "base_delay_s": 0,
    }
    params.update(kw)
    return HttpPageFetcher(client, **params)  # type: ignore[arg-type]


@respx.mock
async def test_extracts_clean_text_and_title() -> None:
    respx.get("https://blog.example/post").respond(
        200, html=ARTICLE, headers={"content-type": "text/html; charset=utf-8"}
    )
    async with httpx.AsyncClient() as client:
        page = await make_fetcher(client).fetch("https://blog.example/post")
    assert page.title == "Observability for LLM apps"
    assert "OpenTelemetry" in page.text
    assert "Home | About" not in page.text
    assert page.url == "https://blog.example/post"


@respx.mock
async def test_truncates_to_max_chars() -> None:
    respx.get("https://blog.example/long").respond(200, text="word " * 5000)
    async with httpx.AsyncClient() as client:
        page = await make_fetcher(client, max_chars=100).fetch("https://blog.example/long")
    assert len(page.text) <= 100 + len(" [truncated]")
    assert page.truncated


@respx.mock
async def test_stops_reading_after_max_bytes() -> None:
    respx.get("https://blog.example/huge").respond(200, text="a" * 50_000)
    async with httpx.AsyncClient() as client:
        page = await make_fetcher(client, max_bytes=1_000).fetch("https://blog.example/huge")
    assert len(page.text) <= 1_000 + len(" [truncated]")
    assert page.truncated


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://127.0.0.1:8000/health",
        "http://localhost/",
        "https://internal.example/admin",
        "http://meta.example/latest/meta-data",
        "http://[::1]/",
    ],
)
async def test_blocks_non_http_and_private_targets(url: str) -> None:
    async def resolver(host: str) -> list[str]:
        if host == "localhost":
            return ["127.0.0.1"]
        return await public_resolver(host)

    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError):
            await make_fetcher(client, resolver=resolver).fetch(url)


@respx.mock
async def test_redirect_to_private_address_is_blocked() -> None:
    respx.get("https://blog.example/r").respond(
        302, headers={"location": "https://internal.example/secret"}
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError, match="not allowed"):
            await make_fetcher(client).fetch("https://blog.example/r")


@respx.mock
async def test_follows_safe_redirects() -> None:
    respx.get("https://blog.example/old").respond(301, headers={"location": "/new"})
    respx.get("https://blog.example/new").respond(200, text="moved content here")
    async with httpx.AsyncClient() as client:
        page = await make_fetcher(client).fetch("https://blog.example/old")
    assert page.url == "https://blog.example/new"
    assert "moved content" in page.text


@respx.mock
async def test_rejects_binary_content() -> None:
    respx.get("https://blog.example/f.pdf").respond(
        200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError, match="content type"):
            await make_fetcher(client).fetch("https://blog.example/f.pdf")


@respx.mock
async def test_5xx_is_retried_then_raised_as_transient() -> None:
    route = respx.get("https://blog.example/down").respond(503)
    async with httpx.AsyncClient() as client:
        with pytest.raises(TransientFetchError):
            await make_fetcher(client).fetch("https://blog.example/down")
    assert route.call_count == 2


@respx.mock
async def test_404_is_not_retried() -> None:
    route = respx.get("https://blog.example/missing").respond(404)
    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError):
            await make_fetcher(client).fetch("https://blog.example/missing")
    assert route.call_count == 1


@respx.mock
async def test_connection_errors_are_transient() -> None:
    respx.get("https://blog.example/x").mock(side_effect=httpx.ConnectError("reset"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(TransientFetchError):
            await make_fetcher(client).fetch("https://blog.example/x")


@respx.mock
async def test_empty_extraction_is_an_error() -> None:
    respx.get("https://blog.example/empty").respond(200, html="<html><body></body></html>")
    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError, match="no readable text"):
            await make_fetcher(client).fetch("https://blog.example/empty")
