# 3. Search provider: Tavily by default, DuckDuckGo (ddgs) as keyless fallback

- Status: accepted
- Date: 2026-09-30

## Context and problem statement

The agent needs web search that is cheap, returns relevant snippets for LLM consumption, and
works for someone cloning the repo without signing up for anything.

## Considered options

1. Tavily (search API designed for LLM agents; free tier; API key).
2. DuckDuckGo via the `ddgs` library (no key; scrapes public endpoints; rate limited).
3. Anthropic server-side `web_search` tool.
4. Google/Bing/Brave APIs.

## Decision outcome

A `SearchProvider` Protocol with two adapters. **Tavily is used when `TAVILY_API_KEY` is set;
otherwise DuckDuckGo is used automatically.** Tests use `FakeSearchProvider`.

- Tavily returns LLM-oriented snippets and is reliable enough for evals.
- ddgs keeps the quickstart keyless. Its rate-limit errors are classified as transient and
  retried with exponential backoff; "no results" becomes an empty list, not an error.
- The server-side Anthropic tool was not chosen because the project's point is to show a
  custom tool-calling loop with our own caching, budget, retries and SSRF-safe fetching, and to
  keep search swappable and testable offline.

### Consequences

- Good: zero-key quickstart; provider swap is a config change; providers are unit tested
  with stub clients.
- Bad: two code paths to maintain; ddgs can break when upstream HTML changes (live test
  `tests/live` catches this).
