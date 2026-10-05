"""fetch_page: download a URL and extract clean, size-limited text.

The URL comes from the model (i.e. indirectly from the web), so the fetcher guards against
SSRF: only http(s), only public IPs, and every redirect hop is re-validated.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from pydantic import BaseModel

from research_agent.tools.errors import FetchError, TransientFetchError
from research_agent.tools.retry import call_with_retries

__all__ = ["FetchedPage", "HttpPageFetcher", "PageFetcher", "system_resolver"]

Resolver = Callable[[str], Awaitable[list[str]]]

_MAX_REDIRECTS = 5
_TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml", "text/markdown")
_TRUNCATED = " [truncated]"
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_USER_AGENT = "research-agent/0.1 (+https://github.com/CardosoGuiVi/research-agent-langgraph)"


class FetchedPage(BaseModel):
    url: str
    title: str
    text: str
    truncated: bool = False


@runtime_checkable
class PageFetcher(Protocol):
    async def fetch(self, url: str) -> FetchedPage: ...


async def system_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


class HttpPageFetcher:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        resolver: Resolver = system_resolver,
        max_bytes: int,
        max_chars: int,
        timeout_s: float,
        attempts: int,
        base_delay_s: float = 0.5,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._timeout_s = timeout_s
        self._attempts = attempts
        self._base_delay_s = base_delay_s

    async def fetch(self, url: str) -> FetchedPage:
        return await call_with_retries(
            lambda: self._fetch_once(url),
            attempts=self._attempts,
            timeout_s=self._timeout_s,
            base_delay_s=self._base_delay_s,
            timeout_error=TransientFetchError,
        )

    async def _fetch_once(self, url: str) -> FetchedPage:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            await self._ensure_allowed(current)
            try:
                async with self._client.stream(
                    "GET",
                    current,
                    follow_redirects=False,
                    headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
                ) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise FetchError("redirect without location")
                        current = urljoin(current, location)
                        continue
                    self._raise_for_status(resp.status_code)
                    content_type = resp.headers.get("content-type", "text/plain").lower()
                    if not content_type.startswith(_TEXT_TYPES):
                        raise FetchError(f"unsupported content type: {content_type.split(';')[0]}")
                    body, over_limit = await self._read_limited(resp)
                    encoding = resp.encoding or "utf-8"
            except httpx.TimeoutException as exc:
                raise TransientFetchError("timeout") from exc
            except httpx.TransportError as exc:
                raise TransientFetchError(f"connection error: {type(exc).__name__}") from exc
            raw = body.decode(encoding, errors="replace")
            return await self._to_page(current, raw, content_type, over_limit)
        raise FetchError("too many redirects")

    async def _ensure_allowed(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise FetchError(f"URL scheme not allowed: {parts.scheme or 'none'}")
        host = parts.hostname
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                addresses = [ipaddress.ip_address(a) for a in await self._resolver(host)]
            except OSError as exc:
                raise FetchError(f"cannot resolve host {host}") from exc
        if not addresses or any(not a.is_global for a in addresses):
            raise FetchError(f"target address not allowed: {host}")

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status == 429 or status >= 500:
            raise TransientFetchError(f"HTTP {status}")
        if status >= 400:
            raise FetchError(f"HTTP {status}")

    async def _read_limited(self, resp: httpx.Response) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        size = 0
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self._max_bytes:
                return b"".join(chunks)[: self._max_bytes], True
        return b"".join(chunks), False

    async def _to_page(self, url: str, raw: str, content_type: str, truncated: bool) -> FetchedPage:
        if "html" in content_type:
            text = await asyncio.to_thread(
                trafilatura.extract, raw, include_comments=False, include_tables=True
            )
            match = _TITLE_RE.search(raw)
            title = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
        else:
            text, title = raw, ""
        text = (text or "").strip()
        if not text:
            raise FetchError("no readable text extracted")
        if len(text) > self._max_chars:
            text, truncated = text[: self._max_chars], True
        if truncated:
            text += _TRUNCATED
        return FetchedPage(url=url, title=title or url, text=text, truncated=truncated)
