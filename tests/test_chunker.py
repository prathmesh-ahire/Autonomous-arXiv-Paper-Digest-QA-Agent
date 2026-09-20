from itertools import pairwise

from arxiv_agent.services.chunker import chunk_paper, chunk_section
from arxiv_agent.state import ParsedPaper, Section


def test_chunk_section_splits_long_text_and_applies_overlap() -> None:
    size, overlap = 100, 15
    text = "\n\n".join(
        f"This is paragraph number {i} with some extra padding text in it."
        for i in range(6)
    )
    section = Section(title="Body", text=text)

    chunks = chunk_section(section, size, overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= size * 1.2
    for prev, nxt in pairwise(chunks):
        assert nxt.text.startswith(prev.text[-overlap:])


def test_chunk_section_hard_splits_a_single_long_run_of_text() -> None:
    size, overlap = 50, 10
    section = Section(title="Body", text="x" * 200)

    chunks = chunk_section(section, size, overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= size * 1.2


def test_chunk_section_char_offsets_are_within_bounds() -> None:
    text = "\n\n".join(f"Paragraph {i} of the section text." for i in range(4))
    section = Section(title="Body", text=text)

    chunks = chunk_section(section, size=40, overlap=5)

    for chunk in chunks:
        assert 0 <= chunk.char_start <= chunk.char_end <= len(text)


def test_chunk_section_empty_text_returns_no_chunks() -> None:
    assert chunk_section(Section(title="Body", text=""), 100, 10) == []


def test_chunk_paper_prefixes_section_title_and_reindexes_globally() -> None:
    parsed = ParsedPaper(
        full_text="irrelevant",
        sections=[
            Section(
                title="Introduction",
                text="A short intro paragraph with enough content to clear the fifty character minimum easily.",
            ),
            Section(
                title="Method",
                text="A short method paragraph with enough content to clear the fifty character minimum too.",
            ),
        ],
    )

    chunks = chunk_paper(parsed, size=1000, overlap=150)

    assert [c.section_title for c in chunks] == ["Introduction", "Method"]
    assert chunks[0].text.startswith("[Introduction] ")
    assert chunks[1].text.startswith("[Method] ")
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_paper_drops_chunks_under_min_chars() -> None:
    parsed = ParsedPaper(
        full_text="irrelevant", sections=[Section(title="Tiny", text="short")]
    )

    assert chunk_paper(parsed, size=1000, overlap=150) == []


def test_chunk_paper_empty_sections_returns_no_chunks() -> None:
    assert chunk_paper(ParsedPaper(full_text=""), size=1000, overlap=150) == []
