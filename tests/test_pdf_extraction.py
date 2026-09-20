from pathlib import Path

from arxiv_agent.services.pdf_parser import (
    clean_text,
    extract_with_pdfplumber,
    extract_with_pymupdf,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_SAMPLE_PAPER = _FIXTURES / "sample_paper.pdf"
_TWO_COLUMN = _FIXTURES / "two_column_test.pdf"


def test_extract_with_pymupdf_reads_pages_in_order() -> None:
    pages, meta = extract_with_pymupdf(_SAMPLE_PAPER)

    assert meta["page_count"] == 8
    assert len(pages) == 8
    assert "A Great Paper on Testing Section Detection" in pages[0]
    assert "Abstract" in pages[0]
    assert "1 Introduction" in pages[1]
    assert "References" in pages[7]
    # Reading order within the page: title, then heading, then body.
    assert (
        pages[0].index("A Great Paper")
        < pages[0].index("Abstract")
        < pages[0].index("This paper studies")
    )


def test_extract_with_pymupdf_orders_two_columns_left_then_right() -> None:
    pages, _ = extract_with_pymupdf(_TWO_COLUMN)

    text = pages[0]
    left_last = text.index("LEFT COLUMN LINE FOUR")
    right_first = text.index("RIGHT COLUMN LINE ONE")
    assert left_last < right_first


def test_extract_with_pymupdf_finds_font_heading_candidates() -> None:
    _, meta = extract_with_pymupdf(_SAMPLE_PAPER)

    assert "Abstract" in meta["font_heading_candidates"]
    assert "1 Introduction" in meta["font_heading_candidates"]


def test_extract_with_pdfplumber_matches_page_count() -> None:
    pages, meta = extract_with_pdfplumber(_SAMPLE_PAPER)

    assert meta["page_count"] == 8
    assert len(pages) == 8
    assert "Introduction" in pages[1]


def test_extraction_length_is_close_between_engines() -> None:
    pymupdf_pages, _ = extract_with_pymupdf(_SAMPLE_PAPER)
    pdfplumber_pages, _ = extract_with_pdfplumber(_SAMPLE_PAPER)

    pymupdf_len = sum(len(p) for p in pymupdf_pages)
    pdfplumber_len = sum(len(p) for p in pdfplumber_pages)

    assert pymupdf_len > 1000
    # The two engines shouldn't diverge wildly on the same simple fixture.
    assert pdfplumber_len > pymupdf_len * 0.5


def test_clean_text_fixes_ligatures_and_hyphen_breaks() -> None:
    raw = "This is a conﬁguration of a demon-\nstration effect."
    assert clean_text(raw) == "This is a configuration of a demonstration effect."


def test_clean_text_strips_page_number_lines() -> None:
    raw = "Some content\n42\nMore content"
    assert clean_text(raw) == "Some content\nMore content"


def test_clean_text_strips_lines_repeated_three_or_more_times() -> None:
    raw = "\n".join(
        ["Running Header", "Body one"]
        + ["Running Header", "Body two"]
        + ["Running Header", "Body three"]
    )
    cleaned = clean_text(raw)
    assert "Running Header" not in cleaned
    assert "Body one" in cleaned and "Body two" in cleaned and "Body three" in cleaned


def test_clean_text_empty_input() -> None:
    assert clean_text("") == ""
