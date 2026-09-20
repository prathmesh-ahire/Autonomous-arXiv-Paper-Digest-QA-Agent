"""Shared pytest fixtures.

Most node/service tests build their own local `_meta()`/`_parsed()` helpers
and stub LLMs (see e.g. `test_summarize.py`, `test_qa.py`) since each needs
slightly different data. These fixtures exist for new tests that just need a
plausible, ready-made `Settings`/`PaperMeta`/`ParsedPaper`/fake LLM and don't
care about the specifics.
"""

from datetime import date
from pathlib import Path

import pytest

from arxiv_agent.config import Settings
from arxiv_agent.state import PaperMeta, ParsedPaper, Section


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """A `Settings` instance with every data path redirected under `tmp_path`,
    so tests never touch the real `./data/` directory."""
    return Settings(
        data_dir=tmp_path,
        pdf_dir=tmp_path / "pdfs",
        chroma_dir=tmp_path / "chroma",
        output_dir=tmp_path / "outputs",
        checkpoint_db=tmp_path / "checkpoints" / "checkpoints.sqlite",
    )


class FakeLLM:
    """`LLMClient` test double: returns a canned response keyed by the first
    matching substring found in the combined system+user prompt.

    Raises `AssertionError` on an unmatched prompt (unless `default` is set),
    so a test's response map doubles as a spec of which prompts it expects.
    """

    name = "fake"
    max_context_tokens = 100_000

    def __init__(self, responses: dict[str, str], default: str | None = None) -> None:
        self._responses = responses
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        haystack = f"{system}\n{user}"
        for substring, response in self._responses.items():
            if substring in haystack:
                return response
        if self._default is not None:
            return self._default
        raise AssertionError(
            f"FakeLLM: no canned response matched prompt: {haystack[:200]!r}"
        )


@pytest.fixture
def fake_llm() -> type[FakeLLM]:
    """The `FakeLLM` class, so a test can construct it with its own
    substring -> response map: `fake_llm({"rank": '{"index": 1, ...}'})`."""
    return FakeLLM


@pytest.fixture
def sample_paper_meta() -> PaperMeta:
    """A plausible `PaperMeta` for tests that just need *some* selected paper."""
    return PaperMeta(
        arxiv_id="2301.00001",
        title="Attention Is a Useful Mechanism for Testing Fixtures",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract=(
            "We study a fixture paper used only in tests. It has an abstract "
            "long enough to look like a real one."
        ),
        published=date(2023, 1, 1),
        categories=["cs.LG"],
        pdf_url="https://arxiv.org/pdf/2301.00001",
        abs_url="https://arxiv.org/abs/2301.00001",
    )


@pytest.fixture
def sample_parsed() -> ParsedPaper:
    """A `ParsedPaper` with 4 known sections, for tests that need parsed
    content but not a real PDF fixture."""
    sections = [
        Section(
            title="Abstract",
            text="This paper studies a fixture mechanism for testing purposes.",
            level=1,
            start_page=0,
        ),
        Section(
            title="1 Introduction",
            text=(
                "Testing agent pipelines end to end is hard without stable "
                "inputs. This paper introduces a fixture mechanism that "
                "removes network and model variance from the test suite."
            ),
            level=1,
            start_page=0,
        ),
        Section(
            title="2 Method",
            text=(
                "The method fabricates a small, deterministic paper with a "
                "known abstract, introduction, method, and conclusion, each "
                "long enough to exercise chunking and summarization."
            ),
            level=1,
            start_page=1,
        ),
        Section(
            title="3 Conclusion",
            text=(
                "We conclude that deterministic fixtures make agent tests "
                "fast and reliable, at the cost of not exercising real PDF "
                "parsing edge cases."
            ),
            level=1,
            start_page=2,
        ),
    ]
    return ParsedPaper(
        full_text="\n\n".join(section.text for section in sections),
        sections=sections,
        references=["[1] Someone, Somewhere. A Cited Paper. 2020."],
        page_count=3,
        parse_method="pymupdf",
        parse_quality=0.95,
    )
