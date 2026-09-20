"""Terminal error-handling node: records a user-facing message and ends the
graph run.

Reached only for genuinely unrecoverable states - an unrecognized intent, or
an internal failure (e.g. embedding) that a node caught and reported via
`state.errors` rather than raising. Interactive follow-ups that already carry
their own context (`next_action` in `{"confirm_with_user",
"ask_user_to_rephrase"}`, set by `select_paper`/`search_arxiv`) end the graph
directly instead of routing here - see `docs/decisions.md`.
"""

from __future__ import annotations

import logging

from arxiv_agent.state import AgentState, Intent

logger = logging.getLogger(__name__)

_UNKNOWN_INTENT_MESSAGE = (
    "I couldn't understand that as an arXiv ID, URL, or topic. Please provide "
    "a paper to look up or a topic to search for."
)
_GENERIC_MESSAGE = "The agent could not complete this request."


def handle_error(state: AgentState) -> dict:
    """Pick a user-facing error message from state and route the run to `END`."""
    if state.errors:
        message = state.errors[-1]
    elif state.intent == Intent.UNKNOWN:
        message = _UNKNOWN_INTENT_MESSAGE
    else:
        message = _GENERIC_MESSAGE

    logger.info("handle_error: %s", message)
    errors = (
        state.errors
        if state.errors and state.errors[-1] == message
        else [
            *state.errors,
            message,
        ]
    )
    return {"errors": errors, "next_action": "error"}
