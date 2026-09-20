"""LLM client factory: picks a provider and wraps it with retry, rate
limiting, and an on-disk response cache."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.llm.gemini import GeminiClient
from arxiv_agent.llm.groq_client import GroqClient
from arxiv_agent.llm.null import NullLLM

logger = logging.getLogger(__name__)


def get_llm(settings: Settings) -> LLMClient:
    """Build the configured `LLMClient`, or `NullLLM` if no matching key is set."""
    base: LLMClient
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        base = GeminiClient(api_key=settings.gemini_api_key, model=settings.llm_model)
    elif settings.llm_provider == "groq" and settings.groq_api_key:
        base = GroqClient(api_key=settings.groq_api_key, model=settings.llm_model)
    else:
        if settings.llm_provider != "none":
            logger.warning(
                "llm_provider=%r but no matching API key is set; falling back to NullLLM",
                settings.llm_provider,
            )
        return NullLLM()

    # Rate limiting wraps the raw client, and retrying wraps *that*, so every
    # real HTTP attempt - including retries after a transient/429 error -
    # consumes a rate-limit token. Wrapping the other way around would let a
    # single logical call that retries 3 times spend 3 real requests against
    # the provider's per-minute quota while only ever consuming 1 token here,
    # silently defeating the "cannot exceed N requests/minute" guarantee this
    # wrapper exists for - see docs/decisions.md.
    client: LLMClient = _RateLimitedClient(
        base, requests_per_minute=settings.llm_requests_per_minute
    )
    client = _RetryingClient(client)
    client = _CachedClient(
        client,
        cache_dir=settings.data_dir / "llm_cache",
        cache_key_prefix=f"{base.name}:{settings.llm_model}",
    )
    return client


class _RetryingClient:
    """Retries transient (`LLMError.retryable`) failures with 1s/2s/4s backoff."""

    _DELAYS = (1, 2, 4)

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        """Delegate to the wrapped client's provider identifier."""
        return self._inner.name

    @property
    def max_context_tokens(self) -> int:
        """Delegate to the wrapped client's context window."""
        return self._inner.max_context_tokens

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Call the wrapped client, retrying retryable failures with backoff."""
        last_exc: LLMError | None = None
        attempts = len(self._DELAYS) + 1
        for attempt in range(attempts):
            if attempt:
                delay = self._DELAYS[attempt - 1]
                logger.warning(
                    "Retrying %s call in %ss (attempt %d/%d) after: %s",
                    self._inner.name,
                    delay,
                    attempt + 1,
                    attempts,
                    last_exc,
                )
                time.sleep(delay)
            try:
                return self._inner.complete(system, user, temperature)
            except LLMError as exc:
                last_exc = exc
                if not exc.retryable:
                    raise
        assert last_exc is not None
        raise last_exc


class _RateLimitedClient:
    """Token-bucket limiter capping requests/minute against `self._inner`."""

    def __init__(self, inner: LLMClient, requests_per_minute: int) -> None:
        self._inner = inner
        self._capacity = max(requests_per_minute, 1)
        self._tokens = float(self._capacity)
        self._refill_per_second = self._capacity / 60.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Delegate to the wrapped client's provider identifier."""
        return self._inner.name

    @property
    def max_context_tokens(self) -> int:
        """Delegate to the wrapped client's context window."""
        return self._inner.max_context_tokens

    def _acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last_refill) * self._refill_per_second,
            )
            self._last_refill = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._refill_per_second
                logger.debug("Rate limit: waiting %.2fs for a token", wait)
                time.sleep(wait)
                self._last_refill = time.monotonic()
                self._tokens = 0
            else:
                self._tokens -= 1

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Wait for a rate-limit token, then call the wrapped client."""
        self._acquire()
        return self._inner.complete(system, user, temperature)


class _CachedClient:
    """On-disk cache keyed by sha256(system + user + model); outermost wrapper
    so a cache hit skips rate limiting and retries entirely."""

    def __init__(
        self, inner: LLMClient, cache_dir: Path, cache_key_prefix: str
    ) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_key_prefix = cache_key_prefix

    @property
    def name(self) -> str:
        """Delegate to the wrapped client's provider identifier."""
        return self._inner.name

    @property
    def max_context_tokens(self) -> int:
        """Delegate to the wrapped client's context window."""
        return self._inner.max_context_tokens

    def _cache_path(self, system: str, user: str) -> Path:
        key_material = f"{self._cache_key_prefix}\x00{system}\x00{user}"
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Return a cached response if present, else call and cache the wrapped client."""
        path = self._cache_path(system, user)
        if path.exists():
            logger.debug("LLM cache hit: %s", path.name)
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        start = time.monotonic()
        response = self._inner.complete(system, user, temperature)
        latency = time.monotonic() - start
        logger.debug(
            "LLM call: provider=%s prompt_chars=%d latency=%.2fs",
            self._inner.name,
            len(system) + len(user),
            latency,
        )
        path.write_text(json.dumps({"response": response}), encoding="utf-8")
        return response
