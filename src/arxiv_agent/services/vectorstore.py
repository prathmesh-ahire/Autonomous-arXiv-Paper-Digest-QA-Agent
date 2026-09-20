"""Local Chroma vector store: one persistent collection per paper, keyed by
arXiv ID, so re-running the same paper reuses its embeddings instead of
recomputing them.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.services.chunker import Chunk

if TYPE_CHECKING:
    from chromadb import ClientAPI

logger = logging.getLogger(__name__)

_INVALID_NAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_NAME_LEN = 63
_MIN_NAME_LEN = 3


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    section_title: str
    chunk_index: int
    score: float


@lru_cache
def _client_for_path(path: str) -> ClientAPI:
    import chromadb

    return chromadb.PersistentClient(path=path)


def get_client(*, settings: Settings | None = None) -> ClientAPI:
    """Return the (cached, per-path) persistent Chroma client."""
    settings = settings or get_settings()
    return _client_for_path(str(settings.chroma_dir))


def collection_name(arxiv_id: str) -> str:
    """Sanitize an arXiv ID into a name Chroma accepts: 3-63 chars,
    alphanumeric/underscore/hyphen only, starting and ending alphanumeric."""
    sanitized = _INVALID_NAME_CHARS_RE.sub("_", arxiv_id)
    name = f"paper_{sanitized}"[:_MAX_NAME_LEN]
    name = re.sub(r"^[^a-zA-Z0-9]+", "", name) or "paper"
    name = re.sub(r"[^a-zA-Z0-9]+$", "", name) or "paper"
    if len(name) < _MIN_NAME_LEN:
        name = (name + "_pad")[:_MAX_NAME_LEN]
    return name


def _get_collection(arxiv_id: str, *, settings: Settings | None = None):
    import chromadb

    client = get_client(settings=settings)
    try:
        return client.get_collection(collection_name(arxiv_id))
    except chromadb.errors.NotFoundError:
        return None


def collection_exists(
    arxiv_id: str,
    expected_count: int | None = None,
    *,
    settings: Settings | None = None,
) -> bool:
    """True if the paper already has a Chroma collection - and, when
    `expected_count` is given, only if its chunk count matches it, so a
    changed chunking strategy doesn't silently reuse a stale collection."""
    collection = _get_collection(arxiv_id, settings=settings)
    if collection is None:
        return False
    if expected_count is None:
        return True
    return collection.count() == expected_count


def upsert_chunks(
    arxiv_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    settings: Settings | None = None,
) -> None:
    """Store each chunk's text, embedding, and metadata in the paper's
    collection, creating it if it doesn't exist yet."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "must be the same length"
        )
    if not chunks:
        return

    client = get_client(settings=settings)
    collection = client.get_or_create_collection(
        collection_name(arxiv_id), metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_estimate": chunk.token_estimate,
            }
            for chunk in chunks
        ],
    )


def search(
    arxiv_id: str,
    query_embedding: list[float],
    top_k: int,
    *,
    where: dict | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return the `top_k` chunks closest to `query_embedding`.

    The collection is created with cosine distance (`hnsw:space: cosine`),
    so similarity is recovered as `1 - distance`: 1.0 for an identical
    vector, 0.0 for an orthogonal one.
    """
    collection = _get_collection(arxiv_id, settings=settings)
    if collection is None:
        return []

    query_kwargs = {"where": where} if where else {}
    result = collection.query(
        query_embeddings=[query_embedding], n_results=top_k, **query_kwargs
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=document,
            section_title=metadata.get("section_title", ""),
            chunk_index=metadata.get("chunk_index", 0),
            score=1.0 - distance,
        )
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]
