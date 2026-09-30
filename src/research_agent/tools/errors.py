"""Tool error hierarchy. `TransientError` marks failures worth retrying."""


class ToolError(Exception):
    """Base class for tool failures surfaced to the agent as tool results."""


class TransientError(ToolError):
    """Timeouts, 429s, 5xx, connection resets: retry with backoff."""


class BudgetExceededError(ToolError):
    """A per-run cap (searches, fetches, tokens) was reached."""


class SearchError(ToolError):
    pass


class TransientSearchError(SearchError, TransientError):
    pass


class FetchError(ToolError):
    pass


class TransientFetchError(FetchError, TransientError):
    pass
