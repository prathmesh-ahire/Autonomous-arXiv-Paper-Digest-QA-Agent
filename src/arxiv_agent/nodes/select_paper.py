"""Selects the single best-matching paper from `state.candidates`.

Runs a fast embedding-based rank over all candidates, narrows to the top 5,
then (if an LLM is configured) asks it to pick the best of those 5 with a
one-sentence justification. Falls back to the pure embedding rank when no
LLM is available or the LLM call fails. If the winning candidate's embedding
score is below the confidence threshold, defers to the user instead of
guessing.
"""

from __future__ import annotations

import logging

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm import get_llm
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.services.ranker import EmbedFn, embed_rank, llm_rank
from arxiv_agent.state import AgentState

logger = logging.getLogger(__name__)

_LLM_RERANK_POOL = 5


def select_paper(
    state: AgentState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """Rank `state.candidates` and set `selected`, or defer to the user when
    confidence is low or the candidate list is empty."""
    settings = settings or get_settings()
    query = state.search_query or state.raw_input

    ranked = embed_rank(query, state.candidates, embed_fn=embed_fn)
    if not ranked:
        return {"next_action": "ask_user_to_rephrase"}

    pool = ranked[:_LLM_RERANK_POOL]
    best_paper, best_score = pool[0]
    reason = "Highest embedding similarity to the query."

    llm = llm or get_llm(settings)
    try:
        index, llm_reason = llm_rank(query, [paper for paper, _ in pool], llm)
        best_paper, best_score = pool[index]
        reason = llm_reason
    except LLMError as exc:
        logger.info(
            "select_paper: LLM ranking unavailable (%s), using embedding rank", exc
        )

    if best_score < settings.selection_confidence_threshold:
        top3 = ranked[:3]
        return {
            "candidates": [paper for paper, _ in top3],
            "selection_scores": [score for _, score in top3],
            "next_action": "confirm_with_user",
        }

    return {
        "selected": best_paper,
        "selection_reason": reason,
        "selection_scores": [score for _, score in pool],
        "next_action": None,
    }
