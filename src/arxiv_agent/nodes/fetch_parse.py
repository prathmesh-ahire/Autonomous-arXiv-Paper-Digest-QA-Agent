"""Fetch-and-parse node: downloads the selected paper's PDF, extracts text
(retrying with the fallback engine if PyMuPDF's output scores poorly),
truncates oversized papers, detects sections, and produces `state.parsed`.

Degrades rather than crashes: a failed download or an unreadable
(scanned/image-only) PDF falls back to an abstract-only `ParsedPaper` so the
rest of the graph (chunk/embed, summarize) still has something to work with.
"""

from __future__ import annotations

import logging

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import PDFDownloadError, PDFParseError
from arxiv_agent.services import pdf_parser
from arxiv_agent.state import AgentState, PaperMeta, ParsedPaper, Section

logger = logging.getLogger(__name__)


def fetch_parse(state: AgentState, *, settings: Settings | None = None) -> dict:
    """Download and parse `state.selected`'s PDF into `state.parsed`,
    degrading to an abstract-only `ParsedPaper` instead of raising on
    download failure or a near-zero-quality (e.g. scanned) parse."""
    settings = settings or get_settings()
    if state.selected is None:
        raise PDFParseError("fetch_parse called with no selected paper")
    meta = state.selected
    warnings = list(state.warnings)

    try:
        pdf_path = pdf_parser.download_pdf(meta, settings.pdf_dir, settings=settings)
    except PDFDownloadError as exc:
        logger.warning("fetch_parse: download failed for %s: %s", meta.arxiv_id, exc)
        warnings.append(exc.user_message)
        return {"parsed": _abstract_only(meta, warnings), "warnings": warnings}

    pages, doc_meta, parse_method, quality = _extract_best(pdf_path)

    if quality <= pdf_parser.NEAR_ZERO_QUALITY:
        warnings.append(
            "PDF text extraction produced almost no usable text (likely a "
            "scanned or image-only PDF); falling back to abstract-only mode."
        )
        return {"parsed": _abstract_only(meta, warnings), "warnings": warnings}

    page_numbers = list(range(len(pages)))
    if len(pages) > settings.max_pdf_pages:
        original_count = doc_meta.get("page_count", len(pages))
        pages, page_numbers = pdf_parser.truncate_pages(pages, settings.max_pdf_pages)
        warnings.append(
            f"Paper has {original_count} pages; truncated to {len(pages)} "
            f"(first {settings.max_pdf_pages} plus any conclusion/results pages)."
        )

    pages = [pdf_parser.clean_text(page) for page in pages]
    sections = pdf_parser.detect_sections(
        pages,
        font_candidates=doc_meta.get("font_heading_candidates"),
        page_numbers=page_numbers,
    )
    body_sections, references = pdf_parser.split_references(sections)
    body_sections = _with_clean_abstract(body_sections, sections, meta, page_numbers)

    parsed = ParsedPaper(
        full_text=pdf_parser.clean_text(
            "\n\n".join(section.text for section in body_sections)
        ),
        sections=body_sections,
        references=references,
        page_count=doc_meta.get("page_count", len(pages)),
        parse_method=parse_method,
        parse_quality=quality,
        warnings=warnings,
    )
    return {"parsed": parsed, "warnings": warnings}


def _extract_best(pdf_path) -> tuple[list[str], dict, str, float]:
    """Run PyMuPDF first; if its quality score is below threshold, also try
    pdfplumber and keep whichever result scored higher."""
    pages, doc_meta = pdf_parser.extract_with_pymupdf(pdf_path)
    parse_method = "pymupdf"
    quality = pdf_parser.assess_quality(pages)

    if quality < pdf_parser.MIN_QUALITY:
        fallback_pages, fallback_meta = pdf_parser.extract_with_pdfplumber(pdf_path)
        fallback_quality = pdf_parser.assess_quality(fallback_pages)
        if fallback_quality > quality:
            logger.info(
                "fetch_parse: pymupdf quality %.2f below threshold, "
                "pdfplumber scored %.2f - using pdfplumber",
                quality,
                fallback_quality,
            )
            pages, doc_meta, quality = fallback_pages, fallback_meta, fallback_quality
            parse_method = "pdfplumber"

    return pages, doc_meta, parse_method, quality


def _with_clean_abstract(
    body_sections: list[Section],
    all_sections: list[Section],
    meta: PaperMeta,
    page_numbers: list[int],
) -> list[Section]:
    """Replace a PDF-detected Abstract section's text with the (cleaner)
    arXiv API abstract, or prepend one if no Abstract section was detected."""
    abstract_text = pdf_parser.extract_abstract(all_sections, meta)
    if not abstract_text:
        return body_sections

    for i, section in enumerate(body_sections):
        if section.title.strip().lower() == "abstract":
            body_sections = list(body_sections)
            body_sections[i] = section.model_copy(update={"text": abstract_text})
            return body_sections

    start_page = page_numbers[0] if page_numbers else 0
    abstract_section = Section(
        title="Abstract", text=abstract_text, level=1, start_page=start_page
    )
    return [abstract_section, *body_sections]


def _abstract_only(meta: PaperMeta, warnings: list[str]) -> ParsedPaper:
    return ParsedPaper(
        full_text=meta.abstract,
        sections=[Section(title="Abstract", text=meta.abstract, level=1, start_page=0)],
        references=[],
        page_count=0,
        parse_method="abstract_only",
        parse_quality=0.0,
        warnings=warnings,
    )
