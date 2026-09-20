import shutil
from datetime import date
from pathlib import Path

import pytest

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import PDFDownloadError
from arxiv_agent.nodes import fetch_parse as fetch_parse_module
from arxiv_agent.nodes.fetch_parse import fetch_parse
from arxiv_agent.state import AgentState, PaperMeta

_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
_SCANNED_PDF = Path(__file__).parent / "fixtures" / "scanned_image_only.pdf"


def _meta(
    arxiv_id: str = "2301.12345", abstract: str = "A clean API abstract."
) -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Great Paper on Testing Section Detection",
        authors=["Jane Doe"],
        abstract=abstract,
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _settings_with_cached_pdf(
    tmp_path: Path, meta: PaperMeta, fixture: Path
) -> Settings:
    """Pre-populate the PDF cache so `download_pdf` short-circuits without a
    network call — this is the "one real cached PDF fixture" the integration
    test (Phase 19.6) asks for."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(fixture, pdf_dir / f"{meta.arxiv_id}.pdf")
    return Settings(pdf_dir=pdf_dir)


@pytest.mark.integration
def test_fetch_parse_builds_parsed_paper_from_cached_pdf(tmp_path: Path) -> None:
    meta = _meta()
    settings = _settings_with_cached_pdf(tmp_path, meta, _FIXTURE_PDF)
    state = AgentState(selected=meta)

    result = fetch_parse(state, settings=settings)

    parsed = result["parsed"]
    assert parsed.parse_method == "pymupdf"
    assert parsed.parse_quality > 0
    assert parsed.page_count == 8
    titles = [s.title for s in parsed.sections]
    assert "1 Introduction" in titles
    assert "References" not in titles  # excluded from chunking (Phase 19.4)
    assert len(parsed.references) == 3
    abstract_section = next(s for s in parsed.sections if s.title == "Abstract")
    assert abstract_section.text == meta.abstract  # API abstract wins over PDF text


def test_fetch_parse_degrades_to_abstract_only_on_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = _meta()
    settings = Settings(pdf_dir=tmp_path)

    def _raise(*args, **kwargs):
        raise PDFDownloadError(
            "network down", user_message="Couldn't download the PDF."
        )

    monkeypatch.setattr(fetch_parse_module.pdf_parser, "download_pdf", _raise)
    state = AgentState(selected=meta)

    result = fetch_parse(state, settings=settings)

    parsed = result["parsed"]
    assert parsed.parse_method == "abstract_only"
    assert parsed.parse_quality == 0.0
    assert parsed.full_text == meta.abstract
    assert "Couldn't download the PDF." in result["warnings"]


def test_fetch_parse_degrades_to_abstract_only_on_scanned_pdf(tmp_path: Path) -> None:
    meta = _meta()
    settings = _settings_with_cached_pdf(tmp_path, meta, _SCANNED_PDF)
    state = AgentState(selected=meta)

    result = fetch_parse(state, settings=settings)

    parsed = result["parsed"]
    assert parsed.parse_method == "abstract_only"
    assert parsed.parse_quality == 0.0
    assert any("scanned or image-only" in w for w in result["warnings"])


def test_fetch_parse_truncates_oversized_paper(tmp_path: Path) -> None:
    meta = _meta()
    settings = _settings_with_cached_pdf(tmp_path, meta, _FIXTURE_PDF)
    settings = settings.model_copy(update={"max_pdf_pages": 3})
    state = AgentState(selected=meta)

    result = fetch_parse(state, settings=settings)

    assert any("truncated" in w for w in result["warnings"])
    # Page 7 (References, kept via the conclusion/results heading rule
    # doesn't apply here) should be dropped; only pages 0-2 remain.
    max_start_page = max(s.start_page for s in result["parsed"].sections)
    assert max_start_page <= 2


def test_fetch_parse_raises_without_a_selected_paper() -> None:
    from arxiv_agent.exceptions import PDFParseError

    with pytest.raises(PDFParseError):
        fetch_parse(AgentState(), settings=Settings())
