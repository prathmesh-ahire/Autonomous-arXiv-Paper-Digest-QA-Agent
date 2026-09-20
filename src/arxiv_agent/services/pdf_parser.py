"""PDF download and parsing: fetches a paper's PDF, extracts per-page text
with two engines (PyMuPDF primary, pdfplumber fallback), scores extraction
quality, detects section boundaries, and splits the abstract/references out
of the body text that goes on to chunking.

Kept as one module (rather than one file per concern) to match the target
repo structure in docs/TODO.md, which lists a single `services/pdf_parser.py`
for all of Part D (Phases 15-19).
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pdfplumber
import pymupdf
import requests

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import PDFDownloadError
from arxiv_agent.state import PaperMeta, Section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download (Phase 15)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 64 * 1024
_LOG_EVERY_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60
_PDF_MAGIC = b"%PDF"

GetFn = Callable[..., requests.Response]


def _pdf_filename(arxiv_id: str) -> str:
    # Legacy IDs contain a "/" (e.g. "cs/0701001") which isn't valid in a filename.
    return arxiv_id.replace("/", "_") + ".pdf"


def download_pdf(
    meta: PaperMeta,
    dest_dir: Path,
    *,
    settings: Settings | None = None,
    get_fn: GetFn | None = None,
) -> Path:
    """Download `meta`'s PDF into `dest_dir`, named `{arxiv_id}.pdf`.

    Returns the cached path unchanged if it already exists. Raises
    `PDFDownloadError` on a network failure, a non-PDF response, or a file
    larger than `settings.max_pdf_size_mb`.
    """
    settings = settings or get_settings()
    get_fn = get_fn or requests.get
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _pdf_filename(meta.arxiv_id)

    if dest_path.exists():
        logger.info("download_pdf: cache hit for %s at %s", meta.arxiv_id, dest_path)
        return dest_path

    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    try:
        response = get_fn(meta.pdf_url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PDFDownloadError(
            f"Failed to download PDF for {meta.arxiv_id} from {meta.pdf_url}: {exc}",
            user_message=(
                f"I couldn't download the PDF for {meta.arxiv_id}. "
                "Check your connection and try again."
            ),
        ) from exc

    tmp_path = dest_path.with_suffix(".pdf.part")
    downloaded = 0
    next_log_at = _LOG_EVERY_BYTES
    try:
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise PDFDownloadError(
                        f"PDF for {meta.arxiv_id} exceeds max size "
                        f"({settings.max_pdf_size_mb} MB)",
                        user_message=(
                            f"The PDF for {meta.arxiv_id} is larger than the "
                            f"{settings.max_pdf_size_mb} MB limit, so I skipped it."
                        ),
                    )
                fh.write(chunk)
                if downloaded >= next_log_at:
                    logger.info(
                        "download_pdf: %s - %d MB downloaded",
                        meta.arxiv_id,
                        downloaded // (1024 * 1024),
                    )
                    next_log_at += _LOG_EVERY_BYTES
    except PDFDownloadError:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        response.close()

    if tmp_path.read_bytes()[:4] != _PDF_MAGIC:
        tmp_path.unlink(missing_ok=True)
        raise PDFDownloadError(
            f"Downloaded file for {meta.arxiv_id} is not a valid PDF (bad magic bytes)",
            user_message=(
                f"The download for {meta.arxiv_id} doesn't look like a valid PDF."
            ),
        )

    tmp_path.replace(dest_path)
    logger.info("download_pdf: saved %s (%d bytes)", dest_path, downloaded)
    return dest_path


# ---------------------------------------------------------------------------
# Text extraction (Phase 16)
# ---------------------------------------------------------------------------

_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_MIN_HEADING_FONT_RATIO = 1.25
_MAX_HEADING_LINE_LEN = 100


def extract_with_pymupdf(path: Path) -> tuple[list[str], dict]:
    """Extract per-page text with PyMuPDF, reading order-corrected for
    two-column layouts, plus doc metadata (page count, title, and heading
    candidates found via the font-size heuristic)."""
    doc = pymupdf.open(path)
    try:
        pages = [_page_text_pymupdf(page) for page in doc]
        doc_meta = {
            "page_count": doc.page_count,
            "title": (doc.metadata or {}).get("title") or None,
            "font_heading_candidates": _font_heading_candidates(doc),
        }
    finally:
        doc.close()
    return pages, doc_meta


def _page_text_pymupdf(page: pymupdf.Page) -> str:
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    ordered = _order_blocks_for_columns(blocks)
    return "\n".join(b[4].strip() for b in ordered)


def _order_blocks_for_columns(blocks: list[tuple]) -> list[tuple]:
    """Sort text blocks by `(y0, x0)`, splitting into a left/right column
    pass first if the blocks' left edges (`x0`) cluster bimodally.

    Left edges, not x-midpoints, are the reliable column signal: within one
    column every line shares (almost) the same `x0` regardless of its
    length, while `x1`/the midpoint swings with each line's text length
    even in single-column prose — clustering on the midpoint misclassifies
    a short heading next to a long paragraph as two different columns.
    """
    by_position = lambda b: (round(b[1], 1), b[0])
    if len(blocks) < 4:
        return sorted(blocks, key=by_position)

    left_edges = [b[0] for b in blocks]
    split = _bimodal_split(left_edges)
    if split is None:
        return sorted(blocks, key=by_position)

    left = sorted(
        (b for b, x0 in zip(blocks, left_edges) if x0 < split), key=by_position
    )
    right = sorted(
        (b for b, x0 in zip(blocks, left_edges) if x0 >= split), key=by_position
    )
    return left + right


def _bimodal_split(values: list[float]) -> float | None:
    """Return the midpoint between two clusters if `values` splits cleanly
    into two groups (a wide gap relative to the overall spread), else None."""
    if len(values) < 4:
        return None
    ordered = sorted(values)
    span = ordered[-1] - ordered[0]
    if span <= 0:
        return None

    gap, gap_index = max(
        ((ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)),
        key=lambda pair: pair[0],
    )
    if gap < span * 0.25:
        return None
    left_count = gap_index + 1
    right_count = len(ordered) - left_count
    if left_count < 2 or right_count < 2:
        return None
    return (ordered[gap_index] + ordered[gap_index + 1]) / 2


def _font_heading_candidates(doc: pymupdf.Document) -> list[str]:
    """Lines whose font size is materially larger than the document's median
    body-text size — candidate section headings for `detect_sections`."""
    lines: list[tuple[float, str]] = []
    for page in doc:
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text or not spans:
                    continue
                size = max(s.get("size", 0.0) for s in spans)
                lines.append((size, text))

    if not lines:
        return []
    median_size = statistics.median(size for size, _ in lines)
    threshold = median_size * _MIN_HEADING_FONT_RATIO

    candidates: list[str] = []
    seen: set[str] = set()
    for size, text in lines:
        if size < threshold or not (3 <= len(text) <= _MAX_HEADING_LINE_LEN):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def extract_with_pdfplumber(path: Path) -> tuple[list[str], dict]:
    """Fallback extractor: no column or font-heading heuristics, just plain
    per-page text, used when PyMuPDF's output scores poorly."""
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
        doc_meta = {
            "page_count": len(pdf.pages),
            "title": (pdf.metadata or {}).get("Title") or None,
            "font_heading_candidates": [],
        }
    return pages, doc_meta


def clean_text(s: str) -> str:
    """Normalize extracted text: fix ligatures and hyphenated line breaks,
    drop standalone page-number lines, and strip lines repeated often enough
    to be a running header/footer.

    The header/footer pass only has enough context to fire once it sees
    several pages' worth of text in one string — calling it on a single page
    is a safe no-op. The pipeline relies on this by cleaning per-page first
    (ligatures/hyphens/page-numbers) and again on the joined document text
    (where repeated headers actually become visible as duplicates).
    """
    if not s:
        return ""

    for ligature, replacement in _LIGATURES.items():
        s = s.replace(ligature, replacement)
    s = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", s)

    lines = s.split("\n")
    line_counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {
        line
        for line, count in line_counts.items()
        if count >= 3 and 3 <= len(line) <= 120
    }

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and (
            stripped in repeated or _PAGE_NUMBER_LINE_RE.fullmatch(stripped)
        ):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Parse quality & fallbacks (Phase 17)
# ---------------------------------------------------------------------------

MIN_QUALITY = 0.4
"""Below this score, `fetch_parse` retries with the other extraction engine."""

NEAR_ZERO_QUALITY = 0.05
"""Below this score even after retrying both engines, treat the PDF as
unreadable (scanned/image-only) and degrade to abstract-only mode."""

_CONCLUSION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(conclusion|conclusions|discussion|results)\b",
    re.IGNORECASE,
)


def assess_quality(pages: list[str]) -> float:
    """Score extracted text 0-1 on chars-per-page, alphabetic ratio, presence
    of the word "abstract", and whitespace ratio (too little OR too much
    whitespace both indicate a bad extraction)."""
    if not pages:
        return 0.0
    full_text = "\n".join(pages)
    if not full_text.strip():
        return 0.0
    total_chars = len(full_text)

    chars_per_page = total_chars / len(pages)
    chars_score = min(chars_per_page / 1500, 1.0)

    alpha_ratio = sum(c.isalpha() for c in full_text) / total_chars

    has_abstract = 1.0 if re.search(r"\babstract\b", full_text, re.IGNORECASE) else 0.0

    whitespace_ratio = sum(c.isspace() for c in full_text) / total_chars
    whitespace_score = max(0.0, 1.0 - abs(whitespace_ratio - 0.18) / 0.18)

    score = (
        0.35 * chars_score
        + 0.35 * alpha_ratio
        + 0.15 * has_abstract
        + 0.15 * whitespace_score
    )
    return round(score, 4)


def truncate_pages(pages: list[str], max_pages: int) -> tuple[list[str], list[int]]:
    """Keep the first `max_pages` pages plus any later page opening with a
    conclusion/results/discussion heading. Returns `(kept_pages,
    original_page_indices)` so downstream code can still report true page
    numbers after pages in between were dropped."""
    if len(pages) <= max_pages:
        return pages, list(range(len(pages)))

    kept_indices = list(range(max_pages))
    for i in range(max_pages, len(pages)):
        if _CONCLUSION_HEADING_RE.search(pages[i][:200]):
            kept_indices.append(i)
    return [pages[i] for i in kept_indices], kept_indices


# ---------------------------------------------------------------------------
# Section detection (Phase 18)
# ---------------------------------------------------------------------------

_NUMBERED_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{0,100})$", re.MULTILINE
)
_CITATION_LIKE_RE = re.compile(r"\bet al\.|\(\d{4}[a-z]?\)|\[\d+\]\s*$")
_MAX_SECTION_NUMBER = 20


def _looks_like_section_number(number: str, title: str) -> bool:
    """Reject numbers the regex would otherwise treat as a heading but that
    are really a results-table score or a bare year sharing a line with
    trailing prose - e.g. "88.3 Petrov et al. (2006) [29]" or "2014 English-
    French dataset ..." from a table extracted as flat text. Real section
    numbers are small and never zero-padded, and their heading text is never
    a citation."""
    parts = number.split(".")
    if int(parts[0]) > _MAX_SECTION_NUMBER:
        return False
    if any(len(p) > 1 and p.startswith("0") for p in parts[1:]):
        return False
    return not _CITATION_LIKE_RE.search(title)


_CANONICAL_HEADINGS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach", "experiments",
    "experimental setup", "results", "discussion", "limitations",
    "conclusion", "conclusions", "references", "appendix", "appendices",
]  # fmt: skip
_KEYWORD_HEADING_RE = re.compile(
    r"(?im)^\s*(" + "|".join(re.escape(h) for h in _CANONICAL_HEADINGS) + r")\s*:?\s*$"
)

_MERGE_MIN_GAP = 3  # chars; candidates this close together are the same line


def _numbered_heading_candidates(text: str) -> list[tuple[int, str, int]]:
    candidates = []
    for m in _NUMBERED_HEADING_RE.finditer(text):
        number, title = m.group(1), m.group(2).strip()
        if not _looks_like_section_number(number, title):
            continue
        level = number.count(".") + 1
        candidates.append((m.start(), f"{number} {title}", level))
    return candidates


def _keyword_heading_candidates(text: str) -> list[tuple[int, str, int]]:
    return [
        (m.start(), m.group(1).strip().title(), 1)
        for m in _KEYWORD_HEADING_RE.finditer(text)
    ]


def _font_heading_position_candidates(
    text: str, font_candidates: list[str]
) -> list[tuple[int, str, int]]:
    candidates = []
    for candidate in font_candidates:
        m = re.search(r"(?m)^\s*" + re.escape(candidate) + r"\s*$", text)
        if m:
            candidates.append((m.start(), candidate.strip(), 1))
    return candidates


def _merge_candidates(
    *groups: list[tuple[int, str, int]],
) -> list[tuple[int, str, int]]:
    """Merge heading candidates from all sources, deduplicating near-identical
    positions (same line found by more than one heuristic) and enforcing
    strictly increasing offsets. Earlier groups win ties, i.e. numbered >
    keyword > font, since callers pass them in that priority order."""
    tagged = [
        (pos, title, level, priority)
        for priority, group in enumerate(groups)
        for pos, title, level in group
    ]
    tagged.sort(key=lambda c: (c[0], c[3]))

    merged: list[tuple[int, str, int]] = []
    last_pos = -_MERGE_MIN_GAP
    for pos, title, level, _priority in tagged:
        if pos - last_pos < _MERGE_MIN_GAP:
            continue
        merged.append((pos, title, level))
        last_pos = pos
    return merged


def _join_pages(pages: list[str]) -> tuple[str, list[int]]:
    """Join pages with a blank-line separator, returning the joined text and
    each page's starting character offset within it."""
    offsets = []
    cursor = 0
    for page in pages:
        offsets.append(cursor)
        cursor += len(page) + 2  # "\n\n" separator added by the join below
    return "\n\n".join(pages), offsets


def _page_for_offset(pos: int, page_offsets: list[int]) -> int:
    page_index = 0
    for i, offset in enumerate(page_offsets):
        if offset <= pos:
            page_index = i
        else:
            break
    return page_index


def _build_sections(
    full_text: str,
    candidates: list[tuple[int, str, int]],
    page_offsets: list[int],
    page_numbers: list[int],
) -> list[Section]:
    sections = []
    for i, (pos, title, level) in enumerate(candidates):
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(full_text)
        # The heading itself is the first line of this slice; drop it so the
        # section's `text` is just its content.
        _heading_line, _, body = full_text[pos:end].partition("\n")
        start_page = page_numbers[_page_for_offset(pos, page_offsets)]
        sections.append(
            Section(title=title, text=body.strip(), level=level, start_page=start_page)
        )
    return sections


def detect_sections(
    pages: list[str],
    *,
    font_candidates: list[str] | None = None,
    page_numbers: list[int] | None = None,
) -> list[Section]:
    """Detect section boundaries by merging three signals: numbered headings
    (`^\\d+(\\.\\d+)*\\s+[A-Z]`), canonical keyword headings (Introduction,
    Method, ...), and font-size-based candidates from `extract_with_pymupdf`.
    Falls back to a single `Section(title="Full Text")` when fewer than 3
    sections come out of that merge (e.g. a plain-text or oddly-formatted
    paper the heuristics can't parse)."""
    if not pages:
        return []
    page_numbers = page_numbers if page_numbers is not None else list(range(len(pages)))
    full_text, page_offsets = _join_pages(pages)

    merged = _merge_candidates(
        _numbered_heading_candidates(full_text),
        _keyword_heading_candidates(full_text),
        _font_heading_position_candidates(full_text, font_candidates or []),
    )

    sections = _build_sections(full_text, merged, page_offsets, page_numbers)
    if len(sections) < 3:
        logger.info(
            "detect_sections: only %d section(s) found, falling back to Full Text",
            len(sections),
        )
        start_page = page_numbers[0] if page_numbers else 0
        return [
            Section(
                title="Full Text",
                text=full_text.strip(),
                level=1,
                start_page=start_page,
            )
        ]
    return sections


# ---------------------------------------------------------------------------
# Abstract, references (Phase 19)
# ---------------------------------------------------------------------------

_REFERENCE_HEADINGS = {"references", "bibliography"}
_APPENDIX_HEADINGS = {"appendix", "appendices"}
_BRACKET_REF_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s*")


def extract_abstract(sections: list[Section], meta: PaperMeta) -> str:
    """Prefer the arXiv API abstract (cleaner than PDF extraction) and fall
    back to a PDF-detected "Abstract" section if the API abstract is empty."""
    if meta.abstract and meta.abstract.strip():
        return meta.abstract.strip()
    for section in sections:
        if section.title.strip().lower() == "abstract":
            return section.text.strip()
    return ""


def split_references(sections: list[Section]) -> tuple[list[Section], list[str]]:
    """Cut the body at the first References/Bibliography/Appendix heading —
    everything from there on is excluded from what goes to chunking (see
    docs/decisions.md). Returns `(body_sections, reference_entries)`."""
    cutoff = len(sections)
    for i, section in enumerate(sections):
        title = section.title.strip().lower()
        if title in _REFERENCE_HEADINGS or title in _APPENDIX_HEADINGS:
            cutoff = i
            break

    body_sections = sections[:cutoff]
    references: list[str] = []
    for section in sections[cutoff:]:
        if section.title.strip().lower() in _REFERENCE_HEADINGS:
            references.extend(_parse_reference_entries(section.text))
    return body_sections, references


def _parse_reference_entries(text: str) -> list[str]:
    """Best-effort split of a References section's raw text into entries,
    preferring `[N]`-style markers and falling back to one entry per line."""
    text = text.strip()
    if not text:
        return []

    parts = _BRACKET_REF_RE.split(text)
    if len(parts) > 2:
        entries = []
        numbers_and_bodies = iter(parts[1:])
        for _number, body in zip(numbers_and_bodies, numbers_and_bodies):
            cleaned = " ".join(body.split())
            if cleaned:
                entries.append(cleaned)
        if entries:
            return entries

    return [line.strip() for line in text.splitlines() if line.strip()]
