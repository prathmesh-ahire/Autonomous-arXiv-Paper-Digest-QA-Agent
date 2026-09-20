import json
from datetime import date

from arxiv_agent.config import Settings
from arxiv_agent.nodes.summarize import build_context, summarize
from arxiv_agent.state import AgentState, PaperMeta, ParsedPaper, Section


def _meta(arxiv_id: str = "2301.00001") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper About Testing",
        authors=["A. Author", "B. Author"],
        abstract="An abstract about testing summarization nodes thoroughly.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _parsed(section_text_len: int = 500, parse_quality: float = 0.9) -> ParsedPaper:
    return ParsedPaper(
        full_text="irrelevant",
        sections=[
            Section(title="Abstract", text="Abstract text. " * 5),
            Section(
                title="Introduction",
                text="Intro sentence. " * (section_text_len // 16 + 1),
            ),
            Section(
                title="Method", text="Method sentence. " * (section_text_len // 17 + 1)
            ),
            Section(
                title="Results", text="Result sentence. " * (section_text_len // 17 + 1)
            ),
            Section(title="Conclusion", text="Conclusion sentence. " * 5),
        ],
        parse_method="pymupdf",
        parse_quality=parse_quality,
    )


_BRIEFING_JSON = json.dumps(
    {
        "title": "Wrong Title The LLM Made Up",
        "authors": ["Someone Else"],
        "arxiv_id": "0000.00000",
        "published": "1999-01-01",
        "link": "https://example.com/wrong",
        "why_it_matters": " ".join(["word"] * 50),
        "problem_statement": "The problem.",
        "method": ["step one", "step two", "step three"],
        "key_results": ["result one", "result two"],
        "limitations": ["a limitation"],
        "followup_questions": ["q1?", "q2?", "q3?"],
        "confidence_notes": None,
    }
)


class _StubLLM:
    name = "stub"
    max_context_tokens = 100_000

    def __init__(self, response: str = _BRIEFING_JSON) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        return self._response


def test_build_context_prioritizes_known_sections_and_respects_budget() -> None:
    parsed = _parsed()
    context = build_context(parsed, budget_chars=2000)

    assert "[Abstract]" in context
    assert "[Introduction]" in context
    assert "[Method]" in context
    assert "[Results]" in context
    assert "[Conclusion]" in context
    assert len(context) <= 2000 + 500  # small slack for headers/markers


def test_summarize_postfills_facts_from_paper_meta() -> None:
    meta = _meta()
    state = AgentState(selected=meta, parsed=_parsed())
    llm = _StubLLM()

    result = summarize(state, settings=Settings(), llm=llm)
    briefing = result["briefing"]

    assert briefing.title == meta.title
    assert briefing.authors == meta.authors
    assert briefing.arxiv_id == meta.arxiv_id
    assert briefing.published == meta.published
    assert briefing.link == meta.abs_url
    assert len(llm.calls) == 1  # single-pass: one reduce call, no map calls


def test_summarize_flags_low_quality_parse_in_confidence_notes() -> None:
    state = AgentState(selected=_meta(), parsed=_parsed(parse_quality=0.1))

    result = summarize(state, settings=Settings(), llm=_StubLLM())

    assert "degraded" in result["briefing"].confidence_notes.lower()


def test_summarize_uses_map_reduce_for_long_papers() -> None:
    # Long sections + a tiny context window forces raw_len > budget * 2.
    parsed = _parsed(section_text_len=20_000)
    state = AgentState(selected=_meta(), parsed=parsed)
    llm = _StubLLM()
    # Force a tiny budget so the long sections trip the map-reduce threshold.
    llm.max_context_tokens = 500

    result = summarize(state, settings=Settings(), llm=llm)

    assert "map-reduce" in result["briefing"].confidence_notes.lower()
    # One map call per non-empty section plus one final reduce call.
    assert len(llm.calls) == len(parsed.sections) + 1
