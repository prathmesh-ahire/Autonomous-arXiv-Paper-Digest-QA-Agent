"""Local sentence-transformers embeddings for paper chunks and QA queries.

Runs entirely offline after the model's first download, so retrieval never
depends on an LLM API key, a rate limit, or (beyond that first run) a
network connection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from arxiv_agent.config import get_settings
from arxiv_agent.exceptions import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

_model_download_logged = False


@lru_cache
def get_embedder() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers embedding model. Loaded
    once per process no matter how many times it's requested."""
    global _model_download_logged
    model_name = get_settings().embedding_model
    if not _model_download_logged:
        logger.info(
            "get_embedder: loading embedding model '%s' - downloads ~80 MB on "
            "first run only, then loads from the local cache",
            model_name,
        )
        _model_download_logged = True

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to load embedding model '{model_name}': {exc}",
            user_message=(
                "I couldn't load the local embedding model. On first run this "
                "needs a network connection to download it; after that, check "
                "available disk space."
            ),
        ) from exc


def embed_texts(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts, L2-normalized so cosine similarity reduces to
    a plain dot product."""
    if not texts:
        return []
    try:
        embedder = get_embedder()
        vectors = embedder.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return vectors.tolist()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to embed {len(texts)} text(s): {exc}",
            user_message="I couldn't compute embeddings for the paper's text.",
        ) from exc


def embed_query(text: str) -> list[float]:
    """Embed a single query string with the same normalization as
    `embed_texts`, so its similarity scores are directly comparable."""
    return embed_texts([text])[0]
