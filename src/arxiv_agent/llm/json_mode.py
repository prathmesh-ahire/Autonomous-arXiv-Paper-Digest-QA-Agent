"""Robust extraction and schema validation of JSON from LLM text responses.

LLMs wrap JSON in code fences, prose, or occasionally truncate it. This module
tolerates that instead of requiring perfectly clean output on the first try.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm.base import LLMClient

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output.

    Tries the fenced/whole-text block as-is first, then falls back to
    brace-matching the outermost `{...}` when the model added stray prose.
    """
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in text: {text[:200]!r}")

    depth = 0
    for i in range(start, len(candidate)):
        if candidate[i] == "{":
            depth += 1
        elif candidate[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON object: {exc}") from exc

    raise ValueError(f"No balanced JSON object found in text: {text[:200]!r}")


def complete_json(
    client: LLMClient,
    system: str,
    user: str,
    model_cls: type[T],
    *,
    temperature: float = 0.2,
) -> T:
    """Call the LLM and validate its response into `model_cls`.

    On extraction or validation failure, re-prompts once with the error
    appended and an explicit instruction to return only valid JSON. Raises
    `LLMError` with the raw response attached if the second attempt also fails.
    """
    response = client.complete(system, user, temperature)
    try:
        return model_cls.model_validate(extract_json(response))
    except (ValueError, ValidationError) as exc:
        retry_user = (
            f"{user}\n\n"
            f"Your previous response failed validation with this error:\n{exc}\n\n"
            "Return ONLY valid JSON matching the required schema. No prose, no code fences."
        )
        response = client.complete(system, retry_user, temperature)
        try:
            return model_cls.model_validate(extract_json(response))
        except (ValueError, ValidationError) as exc2:
            raise LLMError(
                f"LLM response failed JSON validation twice: {exc2}\nRaw response: {response!r}",
                user_message="The AI model returned a response that couldn't be parsed. Try again or switch providers.",
            ) from exc2
