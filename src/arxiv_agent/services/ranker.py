"""Candidate ranking: a fast embedding-based cosine rank, optionally refined
by an LLM pick over the top few candidates.

The embedding step here uses its own lazily-loaded `sentence-transformers`
model rather than `services/embeddings.py` (Part E, not built yet): ranking
runs before a paper is fetched/parsed and only ever compares short
title+abstract strings, never paper chunks. See docs/decisions.md — once
Part E lands, this can switch to the shared embedder if useful.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from arxiv_agent.config import get_settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.llm.json_mode import complete_json
from arxiv_agent.state import PaperMeta

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rank.txt"
_SYSTEM_PROMPT = (
    "You are a research assistant helping a user find the single arXiv paper "
    "that best matches their query."
)

EmbedFn = Callable[[list[str]], list[list[float]]]


class _RankChoice(BaseModel):
    index: int
    reason: str


@lru_cache
def _default_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    return _default_embedder().encode(texts, normalize_embeddings=True).tolist()


def _paper_text(paper: PaperMeta) -> str:
    return f"{paper.title}\n{paper.abstract}"


def embed_rank(
    query: str, candidates: list[PaperMeta], *, embed_fn: EmbedFn | None = None
) -> list[tuple[PaperMeta, float]]:
    """Rank candidates by cosine similarity between the query and each
    paper's title+abstract, most similar first."""
    if not candidates:
        return []

    embed_fn = embed_fn or _default_embed_fn
    texts = [query, *[_paper_text(paper) for paper in candidates]]
    vectors = embed_fn(texts)
    query_vec, paper_vecs = vectors[0], vectors[1:]

    # Embeddings are normalized, so the dot product is the cosine similarity.
    scores = [sum(q * p for q, p in zip(query_vec, vec)) for vec in paper_vecs]
    return sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)


def llm_rank(
    query: str, candidates: list[PaperMeta], llm: LLMClient
) -> tuple[int, str]:
    """Ask the LLM to pick the single most relevant candidate.

    Returns `(index into candidates, one-sentence reason)`. Raises `LLMError`
    if the LLM is unavailable, fails, or picks an out-of-range index.
    """
    numbered = "\n\n".join(
        f"{i + 1}. {paper.title}\n{paper.abstract}"
        for i, paper in enumerate(candidates)
    )
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    user = template.format(query=query, candidates=numbered)

    choice = complete_json(llm, _SYSTEM_PROMPT, user, _RankChoice)
    index = choice.index - 1
    if not 0 <= index < len(candidates):
        raise LLMError(f"LLM rank returned out-of-range index {choice.index}")
    return index, choice.reason
