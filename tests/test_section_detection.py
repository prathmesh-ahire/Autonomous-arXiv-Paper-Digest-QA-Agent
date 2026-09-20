from datetime import date
from pathlib import Path

from arxiv_agent.services.pdf_parser import (
    clean_text,
    detect_sections,
    extract_abstract,
    extract_with_pymupdf,
    split_references,
)
from arxiv_agent.state import PaperMeta, Section

_FIXTURES = Path(__file__).parent / "fixtures"


def _detect_from_fixture(name: str) -> list[Section]:
    pages, meta = extract_with_pymupdf(_FIXTURES / name)
    cleaned = [clean_text(p) for p in pages]
    return detect_sections(cleaned, font_candidates=meta["font_heading_candidates"])


def test_detect_sections_finds_introduction_and_conclusion() -> None:
    sections = _detect_from_fixture("sample_paper.pdf")
    titles = [s.title for s in sections]

    assert "1 Introduction" in titles
    assert "6 Conclusion" in titles
    assert titles.index("1 Introduction") < titles.index("6 Conclusion")


def test_detect_sections_maps_start_page_correctly() -> None:
    sections = _detect_from_fixture("sample_paper.pdf")
    by_title = {s.title: s for s in sections}

    assert by_title["1 Introduction"].start_page == 1
    assert by_title["References"].start_page == 7


def test_detect_sections_strips_heading_line_from_body_text() -> None:
    sections = _detect_from_fixture("sample_paper.pdf")
    intro = next(s for s in sections if s.title == "1 Introduction")

    assert "1 Introduction" not in intro.text
    assert "This is the introduction section" in intro.text


def test_detect_sections_ignores_results_table_rows_misread_as_headings() -> None:
    # Reproduces a real bug found reviewing live QA output on 1706.03762: a
    # results table extracted as flat text ("88.3 Petrov et al. (2006) [29]",
    # "23.75 Deep-Att + PosUnk [39]") was matching the numbered-heading regex
    # and being kept as a "section", fragmenting the real Results section and
    # polluting retrieval with citation-like chunks that outscored real
    # content for generic questions.
    page_text = (
        "1 Introduction\n"
        "This paper introduces the Transformer.\n"
        "6 Results\n"
        "Model BLEU\n"
        "88.3 Petrov et al. (2006) [29]\n"
        "90.4 Zhu et al. (2013) [40]\n"
        "23.75 Deep-Att + PosUnk [39]\n"
        "91.7 Transformer (4 layers)\n"
        "The Transformer generalizes well to other tasks.\n"
        "7 Conclusion\n"
        "We presented the Transformer."
    )
    pages = [page_text]

    sections = detect_sections(pages)

    titles = [s.title for s in sections]
    assert "6 Results" in titles
    assert "7 Conclusion" in titles
    assert not any("et al" in t for t in titles)
    assert not any(t.startswith(("88.", "90.", "91.", "23.")) for t in titles)
    results = next(s for s in sections if s.title == "6 Results")
    assert "Petrov et al." in results.text  # kept as body text, not split out


def test_detect_sections_falls_back_to_full_text_when_no_headings() -> None:
    pages = ["Just some plain prose with no headings at all, page one."] * 2

    sections = detect_sections(pages)

    assert len(sections) == 1
    assert sections[0].title == "Full Text"


def test_detect_sections_empty_pages_returns_empty() -> None:
    assert detect_sections([]) == []


def test_detect_sections_uses_supplied_original_page_numbers() -> None:
    pages = [
        "1 Introduction\nIntro text here.",
        "2 Method\nMethod text here.",
        "3 Results\nResults text here.",
        "4 Conclusion\nConclusion text here.",
    ]

    sections = detect_sections(pages, page_numbers=[0, 5, 6, 9])

    by_title = {s.title: s for s in sections}
    assert by_title["2 Method"].start_page == 5
    assert by_title["4 Conclusion"].start_page == 9


def test_split_references_excludes_references_and_appendix() -> None:
    sections = [
        Section(title="1 Introduction", text="intro"),
        Section(title="2 Method", text="method"),
        Section(
            title="References",
            text="[1] A. Author. Title. 2020.\n[2] B. Author. Other. 2021.",
        ),
        Section(title="Appendix", text="extra material"),
    ]

    body, references = split_references(sections)

    assert [s.title for s in body] == ["1 Introduction", "2 Method"]
    assert references == ["A. Author. Title. 2020.", "B. Author. Other. 2021."]


def test_split_references_no_references_heading_keeps_everything() -> None:
    sections = [Section(title="1 Introduction", text="intro")]

    body, references = split_references(sections)

    assert body == sections
    assert references == []


def test_extract_abstract_prefers_meta_abstract_over_pdf() -> None:
    meta = PaperMeta(
        arxiv_id="2301.12345",
        title="T",
        abstract="Clean API abstract.",
        published=date(2023, 1, 1),
        pdf_url="https://arxiv.org/pdf/2301.12345",
        abs_url="https://arxiv.org/abs/2301.12345",
    )
    sections = [Section(title="Abstract", text="Messy PDF-extracted abstract.")]

    assert extract_abstract(sections, meta) == "Clean API abstract."


def test_extract_abstract_falls_back_to_pdf_when_meta_abstract_empty() -> None:
    meta = PaperMeta(
        arxiv_id="2301.12345",
        title="T",
        abstract="",
        published=date(2023, 1, 1),
        pdf_url="https://arxiv.org/pdf/2301.12345",
        abs_url="https://arxiv.org/abs/2301.12345",
    )
    sections = [Section(title="Abstract", text="Messy PDF-extracted abstract.")]

    assert extract_abstract(sections, meta) == "Messy PDF-extracted abstract."
