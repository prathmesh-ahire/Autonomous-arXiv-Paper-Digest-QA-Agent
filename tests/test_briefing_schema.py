from datetime import date

import pytest
from pydantic import ValidationError

from arxiv_agent.schemas import Briefing


def _valid_kwargs(**overrides) -> dict:
    kwargs = {
        "title": "A Paper",
        "authors": ["A. Author"],
        "arxiv_id": "2301.00001",
        "published": date(2023, 1, 1),
        "link": "https://arxiv.org/abs/2301.00001",
        "why_it_matters": " ".join(["word"] * 50),
        "problem_statement": "The problem.",
        "method": ["step one", "step two", "step three"],
        "key_results": ["result one", "result two"],
        "limitations": ["a limitation"],
        "followup_questions": ["q1?", "q2?", "q3?"],
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_briefing_constructs() -> None:
    briefing = Briefing(**_valid_kwargs())
    assert briefing.title == "A Paper"
    assert briefing.confidence_notes is None


def test_empty_limitations_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(limitations=[]))


def test_why_it_matters_too_short_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(why_it_matters="Too short."))


def test_why_it_matters_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(why_it_matters=" ".join(["word"] * 250)))


def test_method_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(method=["only one"]))


def test_method_above_maximum_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(method=[f"step {i}" for i in range(9)]))


def test_key_results_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(key_results=["only one"]))


def test_followup_questions_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(followup_questions=["only one?"]))
    with pytest.raises(ValidationError):
        Briefing(**_valid_kwargs(followup_questions=[f"q{i}?" for i in range(6)]))
