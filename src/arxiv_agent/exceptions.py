"""Agent exception hierarchy.

Each exception carries a `user_message` — a plain-English string safe to show
directly in the CLI, separate from the technical `str(exc)` used in logs.
"""


class AgentError(Exception):
    """Base class for all agent-raised errors."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class ArxivNotFoundError(AgentError):
    """Raised when an arXiv ID lookup returns no results."""


class AmbiguousQueryError(AgentError):
    """Raised when a topic search cannot be resolved to a single paper confidently."""


class PDFDownloadError(AgentError):
    """Raised when a paper's PDF cannot be downloaded or fails validation."""


class PDFParseError(AgentError):
    """Raised when a downloaded PDF cannot be parsed into usable text."""


class EmbeddingError(AgentError):
    """Raised when text embedding fails (model load, disk, or runtime error)."""


class LLMError(AgentError):
    """Raised when an LLM call fails or returns unusable output.

    `retryable` marks transient failures (429/5xx, connection/timeout errors)
    so the retry wrapper in `llm/__init__.py` knows what's worth retrying.
    """

    def __init__(
        self, message: str, user_message: str | None = None, *, retryable: bool = False
    ) -> None:
        super().__init__(message, user_message)
        self.retryable = retryable
