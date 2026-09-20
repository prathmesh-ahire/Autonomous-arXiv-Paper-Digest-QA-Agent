"""Search node: resolves `state.search_query` into `selected` (paper lookup)
or `candidates` (topic search).

Handles arXiv's zero- and many-result edge cases: a zero-result topic search
gets one broadened retry before giving up; a paper-lookup ID that isn't found
routes to the rephrase prompt instead of raising.
"""

from __future__ import annotations

import logging

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import ArxivNotFoundError
from arxiv_agent.services import arxiv_client
from arxiv_agent.state import AgentState, Intent

logger = logging.getLogger(__name__)


def search_arxiv(state: AgentState, *, settings: Settings | None = None) -> dict:
    """Resolve `state.search_query` into `selected` (paper lookup) or
    `candidates` (topic search), handling the zero/many-result edge cases."""
    settings = settings or get_settings()
    if state.intent == Intent.PAPER_LOOKUP:
        return _lookup(state)
    return _topic_search(state, settings)


def _lookup(state: AgentState) -> dict:
    arxiv_id = state.search_query or state.raw_input
    try:
        paper = arxiv_client.fetch_by_id(arxiv_id)
    except ArxivNotFoundError as exc:
        logger.info("search_arxiv: %s", exc)
        return {
            "errors": [*state.errors, exc.user_message],
            "next_action": "ask_user_to_rephrase",
        }
    return {"selected": paper}


def _topic_search(state: AgentState, settings: Settings) -> dict:
    query = state.search_query or state.raw_input
    candidates = arxiv_client.search(query, max_results=settings.arxiv_max_results)
    warnings = list(state.warnings)

    if not candidates:
        broadened = _broaden_query(query)
        if broadened != query:
            logger.info(
                "search_arxiv: zero results for %r, retrying with %r", query, broadened
            )
            warnings.append(f"No results for '{query}'; broadened to '{broadened}'.")
            candidates = arxiv_client.search(
                broadened, max_results=settings.arxiv_max_results
            )

    if not candidates:
        return {"warnings": warnings, "next_action": "ask_user_to_rephrase"}

    capped = candidates[: settings.arxiv_max_results]
    return {
        "candidates": capped,
        "warnings": warnings,
        "needs_ranking": len(capped) > 1,
    }


def _broaden_query(query: str) -> str:
    """Drop the most specific (last) AND-ed term and strip quotes."""
    terms = [t.strip() for t in query.split(" AND ") if t.strip()]
    if len(terms) > 1:
        terms = terms[:-1]
    broadened = " AND ".join(terms) if terms else query
    return broadened.replace('"', "")
