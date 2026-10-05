# 7. LLMProvider port with an OpenRouter adapter

- Status: accepted (OpenRouter adapter not yet exercised against the live API)
- Date: 2026-10-05

## Context and problem statement

We want to try non-Anthropic models (GLM, DeepSeek, GPT, ...) through OpenRouter without
touching the graph. Nodes already called the LLM through a small Protocol (`structured`,
`tool_step`, `stream_text`), but the research node imported a helper from the Anthropic module,
and the container built `AnthropicLLM` unconditionally. No OpenRouter key exists yet.

## Considered options

1. Keep the existing Protocol (renamed `LLMProvider`) and add an `OpenRouterProvider` adapter.
2. Make the provider return a LangChain `BaseChatModel` and let nodes call it directly.
3. Use LiteLLM or a separate gateway service.

## Decision outcome

Option 1. The Protocol already matches what the graph needs and keeps nodes testable with
`FakeLLM`; option 2 would push provider quirks (structured-output method, tool_choice format,
refusal signals) into every node; option 3 adds a dependency or a service for one more provider.

- `LLM_PROVIDER` (`anthropic` default | `openrouter`) selects the adapter in `container.make_llm`.
- `OpenRouterProvider` uses `langchain-openai`'s `ChatOpenAI` with `base_url` set to OpenRouter's
  OpenAI-compatible endpoint; no hand-written HTTP. The dedicated `langchain-openrouter` package
  (0.x) was considered; `ChatOpenAI` was chosen as the mature 1.x option already compatible with
  our pinned `langchain-core`. Revisit if we need OpenRouter-specific fields (e.g. reasoning).
- Structured output uses native JSON-schema output (`response_format`, non-strict), validated
  with pydantic. A first live eval with `method="function_calling"` had DeepSeek ignore the forced
  tool call in 6 of 9 `analysis` calls. Refusal = `finish_reason == "content_filter"`.
- Config: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (required with `openrouter`, no default model),
  optional `OPENROUTER_MODEL_SMART`, `OPENROUTER_BASE_URL`.
- Provider-neutral helpers (`extract_text`, `extract_usage`) moved to `llm/base.py`; a test fails
  if graph code imports a concrete adapter.

### Consequences

- Default behaviour (Anthropic) is unchanged; no key is needed to run the test suite.
- Not yet covered for OpenRouter (add when a key exists): a `@pytest.mark.live` test, an eval run,
  per-model pricing (cost is reported as `null`), effort/reasoning settings, prompt caching.
