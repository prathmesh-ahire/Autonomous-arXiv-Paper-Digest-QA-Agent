import json
from datetime import date
from pathlib import Path

from arxiv_agent.schemas import Briefing
from arxiv_agent.services.renderer import save_briefing, to_json, to_markdown


def _briefing(**overrides) -> Briefing:
    kwargs = {
        "title": "A Paper About Testing",
        "authors": ["A. Author", "B. Author"],
        "arxiv_id": "2301.00001",
        "published": date(2023, 1, 1),
        "link": "https://arxiv.org/abs/2301.00001",
        "why_it_matters": " ".join(["word"] * 50),
        "problem_statement": "The problem statement.",
        "method": ["step one", "step two", "step three"],
        "key_results": ["result one", "result two"],
        "limitations": ["a limitation"],
        "followup_questions": ["q1?", "q2?", "q3?"],
    }
    kwargs.update(overrides)
    return Briefing(**kwargs)


_EXPECTED_MARKDOWN = (
    "# A Paper About Testing\n"
    "\n"
    "**Authors:** A. Author, B. Author  \n"
    "**arXiv ID:** 2301.00001  \n"
    "**Published:** 2023-01-01  \n"
    "**Link:** https://arxiv.org/abs/2301.00001\n"
    "\n"
    "## Why It Matters\n"
    "\n"
    f"{' '.join(['word'] * 50)}\n"
    "\n"
    "## Problem Statement\n"
    "\n"
    "The problem statement.\n"
    "\n"
    "## Method\n"
    "\n"
    "- step one\n"
    "- step two\n"
    "- step three\n"
    "\n"
    "## Key Results\n"
    "\n"
    "- result one\n"
    "- result two\n"
    "\n"
    "## Limitations\n"
    "\n"
    "- a limitation\n"
    "\n"
    "## Suggested Follow-up Questions\n"
    "\n"
    "- q1?\n"
    "- q2?\n"
    "- q3?\n"
)


def test_to_markdown_matches_expected_snapshot() -> None:
    assert to_markdown(_briefing()) == _EXPECTED_MARKDOWN


def test_to_markdown_surfaces_confidence_notes_near_the_top() -> None:
    md = to_markdown(_briefing(confidence_notes="Based on degraded text."))
    note_index = md.index("Based on degraded text.")
    why_it_matters_index = md.index("## Why It Matters")
    assert note_index < why_it_matters_index


def test_to_json_round_trips_through_briefing_model() -> None:
    briefing = _briefing()
    data = json.loads(to_json(briefing))
    assert data["arxiv_id"] == "2301.00001"
    assert Briefing.model_validate(data) == briefing


def test_save_briefing_writes_both_files(tmp_path: Path) -> None:
    briefing = _briefing()
    md_path, json_path = save_briefing(briefing, tmp_path)

    assert md_path == tmp_path / "2301.00001.md"
    assert json_path == tmp_path / "2301.00001.json"
    assert md_path.read_text(encoding="utf-8") == to_markdown(briefing)
    assert json.loads(json_path.read_text(encoding="utf-8"))["arxiv_id"] == "2301.00001"
