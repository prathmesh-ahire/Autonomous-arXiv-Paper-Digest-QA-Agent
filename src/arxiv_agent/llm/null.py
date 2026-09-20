"""Null LLM client used when no provider is configured.

Lets the rest of the pipeline degrade gracefully: parsing and retrieval
(chunking, embedding, vector search) work fully, with no dependency on this
client at all. `understand_query`'s topic-rewrite and `select_paper`'s LLM
re-rank each call `complete()` directly and catch the resulting `LLMError`
to fall back to a deterministic alternative. Summarization has no substitute
for an LLM and simply degrades to no briefing (`state.briefing` stays
`None`) with a warning. QA is the one path with a real content fallback:
`nodes/qa.py` checks for this client by name (`llm.name == "none"`) and
never calls it at all - instead of a guaranteed `LLMError`, it returns the
top retrieved passages verbatim under a "No LLM configured" header, so a
no-API-key run still gets a grounded, substantive answer instead of a
refusal.
"""

from arxiv_agent.exceptions import LLMError


class NullLLM:
    """Raises `LLMError` on every call. `LLMClient`-compatible by duck typing."""

    name = "none"
    max_context_tokens = 0

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Always raise `LLMError` since no provider is configured."""
        raise LLMError(
            "NullLLM.complete() called with no LLM provider configured",
            user_message=(
                "No LLM API key is configured. Set LLM_PROVIDER and GEMINI_API_KEY or "
                "GROQ_API_KEY in .env to enable summarization and query rewriting."
            ),
        )
