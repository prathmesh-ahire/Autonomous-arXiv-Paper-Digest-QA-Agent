"""The LLM client protocol every provider (and the factory's wrappers) implements."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """A minimal, provider-agnostic chat-completion interface."""

    @property
    def name(self) -> str:
        """Short identifier for the provider, e.g. "gemini", "groq", "none"."""
        ...

    @property
    def max_context_tokens(self) -> int:
        """Approximate context window of the configured model, for prompt budgeting."""
        ...

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send a system+user prompt and return the model's text response."""
        ...
