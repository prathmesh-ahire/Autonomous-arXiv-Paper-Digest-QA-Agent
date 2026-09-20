"""Query-understanding node: classifies intent and, for topic searches,
rewrites the raw input into an arXiv-friendly boolean query.

A recognizable arXiv ID or URL short-circuits straight to `PAPER_LOOKUP`
without ever calling the LLM (Phase 12.2). Everything else is treated as a
topic search and rewritten via the LLM, falling back to a deterministic
stopword-stripping rewrite when no LLM is configured or the call fails.
"""

from __future__ import annotations

import logging

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm import get_llm
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.services.arxiv_client import extract_arxiv_id
from arxiv_agent.state import AgentState, Intent

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "to", "and", "or", "about",
    "with", "using", "based", "via", "into", "is", "are", "what", "how",
}  # fmt: skip

_REWRITE_SYSTEM = (
    "You are a research-librarian assistant that rewrites natural-language "
    "topic requests into precise arXiv API search queries."
)
_REWRITE_USER_TEMPLATE = (
    "Rewrite the following topic into a single arXiv API search query using "
    'the "all:" field prefix and boolean AND/OR operators. Quote multi-word '
    'phrases, e.g. all:"kv cache" AND all:compression. Return ONLY the '
    "query string, with no prose and no code fences.\n\nTopic: {topic}"
)


def understand_query(
    state: AgentState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict:
    """Classify `state.raw_input` and return the `intent` / `search_query` update."""
    text = state.raw_input.strip()
    if not text:
        return {"intent": Intent.UNKNOWN}

    arxiv_id = extract_arxiv_id(text)
    if arxiv_id:
        logger.debug("understand_query: matched arXiv ID %s, skipping LLM", arxiv_id)
        return {"intent": Intent.PAPER_LOOKUP, "search_query": arxiv_id}

    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    query = _rewrite_with_llm(text, llm)
    if query is None:
        query = _fallback_rewrite(text)
        logger.info(
            "understand_query: LLM rewrite unavailable, used deterministic fallback"
        )

    return {"intent": Intent.TOPIC_SEARCH, "search_query": query}


def _rewrite_with_llm(topic: str, llm: LLMClient) -> str | None:
    try:
        response = llm.complete(
            _REWRITE_SYSTEM, _REWRITE_USER_TEMPLATE.format(topic=topic), temperature=0.0
        )
    except LLMError as exc:
        logger.debug("understand_query: LLM rewrite failed: %s", exc)
        return None
    query = response.strip().strip("`").strip()
    return query or None


def _fallback_rewrite(topic: str) -> str:
    """Deterministic rewrite: strip stopwords, quote bigrams, join with AND."""
    words = [w.strip(".,;:!?\"'()").lower() for w in topic.split()]
    words = [w for w in words if w and w not in _STOPWORDS]
    if not words:
        return f'all:"{topic.strip()}"'

    phrases = [" ".join(words[i : i + 2]) for i in range(0, len(words), 2)]
    terms = [
        f'all:"{phrase}"' if " " in phrase else f"all:{phrase}" for phrase in phrases
    ]
    return " AND ".join(terms)
