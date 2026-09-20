"""Gemini client implementing the `LLMClient` protocol, via Google AI Studio."""

import logging

from google import genai
from google.genai import errors as genai_errors
from google.genai.types import GenerateContentConfig

from arxiv_agent.exceptions import LLMError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Approximate context windows; unrecognized models fall back to a conservative default.
_MAX_CONTEXT_TOKENS = {
    "gemini-3.6-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
}
_DEFAULT_MAX_CONTEXT_TOKENS = 128_000


class GeminiClient:
    """`LLMClient` implementation backed by Google AI Studio's Gemini API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        """Provider identifier used in logs and cache keys."""
        return "gemini"

    @property
    def max_context_tokens(self) -> int:
        """Approximate context window for the configured model."""
        return _MAX_CONTEXT_TOKENS.get(self.model, _DEFAULT_MAX_CONTEXT_TOKENS)

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send a system+user prompt to Gemini and return its text response."""
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=GenerateContentConfig(
                    system_instruction=system, temperature=temperature
                ),
            )
        except genai_errors.APIError as exc:
            raise LLMError(
                f"Gemini API error ({exc.code}): {exc}",
                user_message="The Gemini API returned an error. Check your API key and rate limits.",
                retryable=exc.code in _RETRYABLE_STATUS_CODES,
            ) from exc

        if not response.text:
            raise LLMError(
                "Gemini returned an empty response",
                user_message="Gemini returned an empty response. Try rephrasing or retrying.",
            )
        return response.text
