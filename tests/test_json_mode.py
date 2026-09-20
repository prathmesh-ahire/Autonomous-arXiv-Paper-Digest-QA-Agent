import pytest
from pydantic import BaseModel

from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm.json_mode import complete_json, extract_json


class _Answer(BaseModel):
    value: int


class _StubClient:
    """A fake LLMClient returning canned responses in order, one per call."""

    name = "stub"
    max_context_tokens = 1000

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        response = self._responses[self.calls]
        self.calls += 1
        return response


# -- extract_json: 5 fixtures (fenced, prose-prefixed, trailing comma, truncated, valid) --


def test_extract_json_valid():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = '```json\n{"a": 1, "b": [1, 2]}\n```'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_prose_prefixed():
    text = 'Sure, here is the result:\n{"a": 1}\nLet me know if you need more.'
    assert extract_json(text) == {"a": 1}


def test_extract_json_trailing_comma_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": 1,}')


def test_extract_json_truncated_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": 1, "b": [1, 2')


# -- complete_json: re-prompt-once and final-failure behavior --


def test_complete_json_succeeds_first_try():
    client = _StubClient(['{"value": 42}'])
    result = complete_json(client, "system", "user", _Answer)
    assert result == _Answer(value=42)
    assert client.calls == 1


def test_complete_json_recovers_after_one_reprompt():
    client = _StubClient(["not json at all", '{"value": 7}'])
    result = complete_json(client, "system", "user", _Answer)
    assert result == _Answer(value=7)
    assert client.calls == 2


def test_complete_json_raises_llm_error_after_two_failures():
    client = _StubClient(["garbage", "still garbage"])
    with pytest.raises(LLMError):
        complete_json(client, "system", "user", _Answer)
    assert client.calls == 2
