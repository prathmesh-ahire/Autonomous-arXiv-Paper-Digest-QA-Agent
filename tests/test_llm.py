import time

import pytest

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm import _CachedClient, _RateLimitedClient, _RetryingClient, get_llm
from arxiv_agent.llm.null import NullLLM


class _StubClient:
    """A fake LLMClient that replays a scripted sequence of results/exceptions."""

    name = "stub"
    max_context_tokens = 1000

    def __init__(self, effects: list) -> None:
        self._effects = list(effects)
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls += 1
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def test_get_llm_returns_null_when_provider_is_none():
    assert isinstance(get_llm(Settings(llm_provider="none")), NullLLM)


def test_get_llm_falls_back_to_null_when_key_missing():
    assert isinstance(
        get_llm(Settings(llm_provider="gemini", gemini_api_key=None)), NullLLM
    )


def test_null_llm_raises_llm_error():
    with pytest.raises(LLMError):
        NullLLM().complete("system", "user")


def test_retrying_client_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    stub = _StubClient([LLMError("rate limited", retryable=True), "ok"])
    client = _RetryingClient(stub)

    assert client.complete("s", "u") == "ok"
    assert stub.calls == 2


def test_retrying_client_does_not_retry_non_transient_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    stub = _StubClient([LLMError("bad request", retryable=False), "ok"])
    client = _RetryingClient(stub)

    with pytest.raises(LLMError):
        client.complete("s", "u")
    assert stub.calls == 1


def test_retrying_client_gives_up_after_all_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    stub = _StubClient([LLMError("down", retryable=True)] * 4)
    client = _RetryingClient(stub)

    with pytest.raises(LLMError):
        client.complete("s", "u")
    assert stub.calls == 4  # 1 initial + 3 retries


def test_rate_limited_client_passes_through_without_blocking():
    stub = _StubClient(["ok"])
    client = _RateLimitedClient(stub, requests_per_minute=1000)

    assert client.complete("s", "u") == "ok"
    assert stub.calls == 1


def test_cached_client_skips_inner_call_on_cache_hit(tmp_path):
    stub = _StubClient(["first", "second"])
    client = _CachedClient(stub, cache_dir=tmp_path, cache_key_prefix="test")

    assert client.complete("s", "u") == "first"
    assert client.complete("s", "u") == "first"  # cache hit: inner not called again
    assert stub.calls == 1


def test_cached_client_distinguishes_prompts(tmp_path):
    stub = _StubClient(["first", "second"])
    client = _CachedClient(stub, cache_dir=tmp_path, cache_key_prefix="test")

    assert client.complete("s", "u1") == "first"
    assert client.complete("s", "u2") == "second"
    assert stub.calls == 2
