"""Section-aware chunking: splits a parsed paper's sections into overlapping
text chunks sized for embedding.

Splits on paragraph boundaries first, falling back to sentences and then raw
character splits for any single piece that alone exceeds the target size, and
carries the tail of each chunk forward into the next as overlap so retrieval
doesn't lose context at a chunk boundary.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from arxiv_agent.state import ParsedPaper, Section

_MIN_CHUNK_CHARS = 50
_CHARS_PER_TOKEN = 4  # rough estimate; avoids a tokenizer dependency

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class Chunk(BaseModel):
    id: str
    text: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    token_estimate: int


def _slugify(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-") or "section"


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    # PDF-extracted section text is usually one block per line rather than
    # blank-line-separated prose (see services/pdf_parser.py), so fall back
    # to single newlines when there's no blank-line structure to split on.
    return [p.strip() for p in text.split("\n") if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _atomic_pieces(text: str, size: int) -> list[str]:
    """Break `text` into pieces no longer than `size`, preferring paragraph,
    then sentence, then raw character boundaries."""
    pieces: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= size:
                pieces.append(sentence)
            else:
                pieces.extend(_hard_split(sentence, size))
    return pieces


def _locate_pieces(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Find each piece's `(start, end)` offset in `text`, scanning forward
    since pieces are produced in document order."""
    spans = []
    cursor = 0
    for piece in pieces:
        idx = text.find(piece, cursor)
        if idx == -1:
            idx = max(cursor, 0)
        spans.append((idx, idx + len(piece)))
        cursor = idx + len(piece)
    return spans


def chunk_section(section: Section, size: int, overlap: int) -> list[Chunk]:
    """Split `section.text` into overlapping chunks of roughly `size`
    characters. Each chunk after the first is seeded with the last `overlap`
    characters of the previous chunk's text.

    Chunk boundaries are tracked as offsets into `section.text` and the
    chunk's `text` is always an exact slice of the source (`text[start:end]`)
    rather than a re-join of its pieces, so no separator characters are
    invented and `char_start`/`char_end` stay exact.
    """
    text = section.text
    pieces = _atomic_pieces(text, size)
    if not pieces:
        return []
    spans = _locate_pieces(text, pieces)

    chunks: list[Chunk] = []
    start = spans[0][0]
    end = start

    def flush(chunk_end: int) -> None:
        chunk_text = text[start:chunk_end].strip()
        if not chunk_text:
            return
        chunks.append(
            Chunk(
                id=f"{_slugify(section.title)}-{len(chunks)}",
                text=chunk_text,
                section_title=section.title,
                chunk_index=len(chunks),
                char_start=start,
                char_end=chunk_end,
                token_estimate=max(1, len(chunk_text) // _CHARS_PER_TOKEN),
            )
        )

    for _piece_start, p_end in spans:
        if end > start and p_end - start > size:
            flush(end)
            start = max(end - overlap, 0) if overlap > 0 else end
        end = p_end

    flush(end)
    return chunks


def chunk_paper(parsed: ParsedPaper, *, size: int, overlap: int) -> list[Chunk]:
    """Chunk every section of a parsed paper into one flat, document-ordered
    list. Each chunk's text is prefixed with `"[{section_title}] "` so
    section context survives into the embedding. Chunks shorter than
    `_MIN_CHUNK_CHARS` (after the prefix) are dropped."""
    chunks: list[Chunk] = []
    for section in parsed.sections:
        for chunk in chunk_section(section, size, overlap):
            prefixed = f"[{section.title}] {chunk.text}"
            if len(prefixed) < _MIN_CHUNK_CHARS:
                continue
            index = len(chunks)
            chunks.append(
                chunk.model_copy(
                    update={
                        "id": f"{_slugify(section.title)}-{index}",
                        "text": prefixed,
                        "chunk_index": index,
                    }
                )
            )
    return chunks
