from pathlib import Path

from arxiv_agent.services.pdf_parser import (
    MIN_QUALITY,
    NEAR_ZERO_QUALITY,
    assess_quality,
    extract_with_pymupdf,
    truncate_pages,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def test_assess_quality_good_fixture_scores_above_threshold() -> None:
    pages, _ = extract_with_pymupdf(_FIXTURES / "sample_paper.pdf")

    assert assess_quality(pages) >= MIN_QUALITY


def test_assess_quality_scanned_fixture_scores_near_zero() -> None:
    pages, _ = extract_with_pymupdf(_FIXTURES / "scanned_image_only.pdf")

    assert assess_quality(pages) <= NEAR_ZERO_QUALITY


def test_assess_quality_empty_pages_returns_zero() -> None:
    assert assess_quality([]) == 0.0
    assert assess_quality(["", ""]) == 0.0


def test_assess_quality_penalizes_sparse_whitespace_heavy_text() -> None:
    dense_good_text = "word " * 400
    sparse_bad_text = "x" + " " * 2000
    assert assess_quality([dense_good_text]) > assess_quality([sparse_bad_text])


def test_truncate_pages_noop_when_under_limit() -> None:
    pages = ["one", "two", "three"]

    kept, indices = truncate_pages(pages, max_pages=5)

    assert kept == pages
    assert indices == [0, 1, 2]


def test_truncate_pages_keeps_first_n_plus_conclusion_pages() -> None:
    pages = [f"page {i}" for i in range(10)]
    pages[8] = "5 Conclusion\nWe conclude the paper here."

    kept, indices = truncate_pages(pages, max_pages=3)

    assert indices == [0, 1, 2, 8]
    assert kept == [pages[0], pages[1], pages[2], pages[8]]


def test_truncate_pages_drops_pages_with_no_conclusion_heading() -> None:
    pages = [f"page {i}" for i in range(6)]

    kept, indices = truncate_pages(pages, max_pages=2)

    assert indices == [0, 1]
    assert kept == [pages[0], pages[1]]
