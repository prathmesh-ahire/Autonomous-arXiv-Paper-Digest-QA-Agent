"""Groq client implementing the `LLMClient` protocol."""

import logging

import groq

from arxiv_agent.exceptions import LLMError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Approximate context windows; unrecognized models fall back to a conservative default.
_MAX_CONTEXT_TOKENS = {
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
}
_DEFAULT_MAX_CONTEXT_TOKENS = 32_000


class GroqClient:
    """`LLMClient` implementation backed by the Groq API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = groq.Groq(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        """Provider identifier used in logs and cache keys."""
        return "groq"

    @property
    def max_context_tokens(self) -> int:
        """Approximate context window for the configured model."""
        return _MAX_CONTEXT_TOKENS.get(self.model, _DEFAULT_MAX_CONTEXT_TOKENS)

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send a system+user prompt to Groq and return its text response."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
        except (groq.APIConnectionError, groq.APITimeoutError) as exc:
            raise LLMError(
                f"Groq connection error: {exc}",
                user_message="Couldn't reach the Groq API. Check your network connection.",
                retryable=True,
            ) from exc
        except groq.APIStatusError as exc:
            raise LLMError(
                f"Groq API error ({exc.status_code}): {exc}",
                user_message="The Groq API returned an error. Check your API key and rate limits.",
                retryable=exc.status_code in _RETRYABLE_STATUS_CODES,
            ) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError(
                "Groq returned an empty response",
                user_message="Groq returned an empty response. Try rephrasing or retrying.",
            )
        return content
