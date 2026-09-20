"""The structured executive briefing schema — the agent's primary output.

Validated strictly (word counts, list lengths) so a malformed LLM response is
caught by `llm.json_mode.complete_json`'s retry-then-raise flow rather than
silently reaching the user.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class Briefing(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    arxiv_id: str
    published: date
    link: str

    why_it_matters: str
    problem_statement: str
    method: list[str] = Field(min_length=3, max_length=8)
    key_results: list[str] = Field(min_length=2)
    # Required and non-empty: the brief explicitly says don't let the model
    # skip this, so there is no default and no upper bound.
    limitations: list[str] = Field(min_length=1)
    followup_questions: list[str] = Field(min_length=3, max_length=5)
    confidence_notes: str | None = None

    @field_validator("why_it_matters")
    @classmethod
    def _why_it_matters_word_count(cls, value: str) -> str:
        word_count = len(value.split())
        if not 40 <= word_count <= 200:
            raise ValueError(f"why_it_matters must be 40-200 words, got {word_count}")
        return value
