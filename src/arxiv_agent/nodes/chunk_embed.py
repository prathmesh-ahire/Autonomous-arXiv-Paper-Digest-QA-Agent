"""Chunk-and-embed node: turns `state.parsed` into a searchable Chroma
collection.

The collection (keyed by arXiv ID) is the handoff artifact between
summarization and the QA loop - see docs/decisions.md. Re-running the same
paper is close to instant: if a collection already exists with the expected
chunk count, embedding is skipped entirely.
"""

from __future__ import annotations

import logging
import time

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import EmbeddingError
from arxiv_agent.services import embeddings, vectorstore
from arxiv_agent.services.chunker import Chunk, chunk_paper
from arxiv_agent.services.embeddings import EmbedFn
from arxiv_agent.state import AgentState, ParsedPaper

logger = logging.getLogger(__name__)


def _abstract_chunk(parsed: ParsedPaper) -> Chunk:
    text = f"[Abstract] {parsed.full_text}".strip()
    return Chunk(
        id="abstract-0",
        text=text,
        section_title="Abstract",
        chunk_index=0,
        char_start=0,
        char_end=len(parsed.full_text),
        token_estimate=max(1, len(text) // 4),
    )


def chunk_embed(
    state: AgentState,
    *,
    settings: Settings | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """Chunk `state.parsed`, embed it, and upsert it into the paper's Chroma
    collection (or reuse an existing one with a matching chunk count)."""
    settings = settings or get_settings()
    embed_fn = embed_fn or embeddings.embed_texts

    if state.selected is None:
        raise EmbeddingError("chunk_embed called with no selected paper")
    if state.parsed is None:
        raise EmbeddingError("chunk_embed called with no parsed paper")

    arxiv_id = state.selected.arxiv_id
    parsed = state.parsed
    degraded = parsed.parse_method == "abstract_only"

    chunks = (
        []
        if degraded
        else chunk_paper(
            parsed, size=settings.chunk_size, overlap=settings.chunk_overlap
        )
    )
    if not chunks:
        chunks = [_abstract_chunk(parsed)]
        degraded = True

    name = vectorstore.collection_name(arxiv_id)
    if vectorstore.collection_exists(arxiv_id, len(chunks), settings=settings):
        logger.info(
            "chunk_embed: %s - collection already has %d chunks, skipping re-embedding",
            arxiv_id,
            len(chunks),
        )
        return {
            "collection_name": name,
            "chunk_count": len(chunks),
            "degraded_retrieval": degraded,
        }

    start = time.monotonic()
    vectors = embed_fn([chunk.text for chunk in chunks])
    vectorstore.upsert_chunks(arxiv_id, chunks, vectors, settings=settings)
    elapsed = time.monotonic() - start

    mean_len = sum(len(chunk.text) for chunk in chunks) / len(chunks)
    logger.info(
        "chunk_embed: %s - %d chunks, mean length %.0f chars, %.2fs",
        arxiv_id,
        len(chunks),
        mean_len,
        elapsed,
    )

    return {
        "collection_name": name,
        "chunk_count": len(chunks),
        "degraded_retrieval": degraded,
    }
