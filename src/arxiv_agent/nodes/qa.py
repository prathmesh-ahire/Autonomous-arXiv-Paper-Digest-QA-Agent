"""QA node: retrieval-augmented question answering over a single paper's
chunk collection.

Retrieval embeds the question (resolving short pronoun-only follow-ups
against the previous question first), searches the paper's Chroma
collection, drops anything below `settings.min_similarity`, deduplicates
near-identical chunks that overlap regions produce, and hands the LLM the
survivors in document order rather than score order so it reads the paper
the way a person would.

Grounding is enforced two ways, not just by prompt wording: a pre-LLM guard
refuses outright (no LLM call at all) when nothing clears the similarity
floor, and a post-LLM guard treats a literal `NOT_IN_PAPER` response the same
way. Citations like `[2]` in a grounded answer are mapped back to the real
`(section_title, chunk_index)` they refer to for display.

When no LLM is configured (`NullLLM`), the LLM is never called at all: the
retrieved passages are returned verbatim as the answer instead of a
guaranteed `LLMError`, so a no-API-key run still gives a substantive,
grounded-in-the-actual-text result. An LLM failure that isn't the "no
provider configured" case (rate limit, network) still refuses, but now
surfaces the retrieved chunks via `/sources` rather than an empty list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm import get_llm
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.services import embeddings, vectorstore
from arxiv_agent.services.embeddings import EmbedFn
from arxiv_agent.services.vectorstore import RetrievedChunk
from arxiv_agent.state import AgentState, QATurn

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "qa.txt"
_SYSTEM_PROMPT = (
    "You are a precise research assistant who answers questions about a "
    "single arXiv paper using only the numbered passages you are given, and "
    "never from outside knowledge."
)

NOT_IN_PAPER = "NOT_IN_PAPER"
_REFUSAL_MESSAGE = "I couldn't find that in this paper."
_NO_LLM_HEADER = (
    "No LLM configured — showing the most relevant passages from the paper."
)

_DEDUPE_PREFIX_LEN = 120
_HISTORY_TURNS = 3
_MAX_STORED_TURNS = 20  # caps state.messages growth over a long QA session
_PRONOUN_RE = re.compile(r"\b(it|this|that|they)\b", re.IGNORECASE)
_PRONOUN_WORD_LIMIT = 8
_CITATION_RE = re.compile(r"\[(\d+)\]")


def resolve_pronoun_reference(question: str, messages: list[QATurn]) -> str:
    """Rewrite a short pronoun-only follow-up ("what about it?") into a
    self-contained question by folding in the previous question, so the
    embedding actually captures what the pronoun refers to."""
    if not messages:
        return question
    word_count = len(question.strip().split())
    if word_count >= _PRONOUN_WORD_LIMIT or not _PRONOUN_RE.search(question):
        return question
    previous_question = messages[-1].question
    return f"{question.rstrip('?.! ')} - referring to: {previous_question}"


def _normalize_prefix(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:_DEDUPE_PREFIX_LEN]


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks whose normalized text prefix matches one already kept,
    keeping the higher-scoring of the pair. Chunk overlap means adjacent
    chunks can share enough leading text to look like duplicate hits."""
    best_by_prefix: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = _normalize_prefix(chunk.text)
        current = best_by_prefix.get(key)
        if current is None or chunk.score > current.score:
            best_by_prefix[key] = chunk
    return list(best_by_prefix.values())


def retrieve(
    state: AgentState,
    question: str,
    top_k: int | None = None,
    *,
    settings: Settings | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[RetrievedChunk]:
    """Embed `question` and return the paper's surviving, deduplicated
    chunks in document order (not score order), ready to hand to the LLM."""
    settings = settings or get_settings()
    embed_fn = embed_fn or embeddings.embed_texts
    top_k = top_k or settings.top_k

    if state.selected is None:
        return []

    resolved = resolve_pronoun_reference(question, state.messages)
    query_vector = embed_fn([resolved])[0]
    results = vectorstore.search(
        state.selected.arxiv_id, query_vector, top_k, settings=settings
    )

    survivors = [c for c in results if c.score >= settings.min_similarity]
    ordered = sorted(_dedupe(survivors), key=lambda c: c.chunk_index)

    for chunk in ordered:
        logger.debug(
            "qa retrieve: section=%r chunk_index=%d score=%.3f",
            chunk.section_title,
            chunk.chunk_index,
            chunk.score,
        )
    return ordered


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({chunk.section_title}) {chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


def _format_history(messages: list[QATurn]) -> str:
    recent = messages[-_HISTORY_TURNS:]
    if not recent:
        return "(no prior questions)"
    return "\n\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in recent)


def _sources_for(answer_text: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Map each `[N]` citation in `answer_text` back to the real
    `(section_title, chunk_index)` pair it refers to, in citation order."""
    cited_numbers = sorted({int(n) for n in _CITATION_RE.findall(answer_text)})
    sources = []
    for number in cited_numbers:
        if 1 <= number <= len(chunks):
            chunk = chunks[number - 1]
            sources.append(
                {
                    "passage": number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                }
            )
    return sources


def _refusal_turn(question: str, chunks: list[RetrievedChunk], message: str) -> QATurn:
    return QATurn(
        question=question,
        answer=message,
        chunk_ids=[c.chunk_id for c in chunks],
        scores=[c.score for c in chunks],
        grounded=False,
    )


def _sources_from_chunks(chunks: list[RetrievedChunk]) -> list[dict]:
    """Build a `{passage, section_title, chunk_index, score}` entry for every
    retrieved chunk, numbered the same way `_format_passages` numbers them.

    Unlike `_sources_for` (which maps only the `[N]` citations an LLM actually
    used), this surfaces everything retrieved - for the no-LLM raw-passage
    turn, and for an LLM-failure refusal, where nothing was cited but the
    chunks that were found are still useful to show via `/sources`."""
    return [
        {
            "passage": i,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score,
        }
        for i, chunk in enumerate(chunks, start=1)
    ]


def _raw_passages_turn(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[QATurn, list[dict]]:
    """Build a QA turn that surfaces the retrieved passages verbatim instead
    of calling the LLM - used when no LLM is configured (`NullLLM`), so a
    no-key run gives a substantive, grounded-in-the-actual-text result rather
    than a guaranteed `LLMError` refusal."""
    answer = f"{_NO_LLM_HEADER}\n\n{_format_passages(chunks)}"
    turn = QATurn(
        question=question,
        answer=answer,
        chunk_ids=[c.chunk_id for c in chunks],
        scores=[c.score for c in chunks],
        grounded=True,
    )
    return turn, _sources_from_chunks(chunks)


def answer_question(
    state: AgentState,
    question: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embed_fn: EmbedFn | None = None,
) -> tuple[QATurn, list[dict]]:
    """Answer one QA turn, grounded strictly in the paper's retrieved chunks.

    Returns the `QATurn` to append to `state.messages` and a list of
    `{passage, section_title, chunk_index, score}` source dicts for display
    (empty only when nothing was retrieved at all - every other outcome,
    including a refusal, surfaces whatever chunks were actually retrieved).
    Never calls the LLM when nothing clears `settings.min_similarity`, or when
    no LLM is configured at all (`NullLLM`) - the latter returns the retrieved
    passages verbatim instead of a guaranteed `LLMError`. Treats a literal
    `NOT_IN_PAPER` response from the LLM the same way as the pre-LLM refusal.
    """
    settings = settings or get_settings()

    chunks = retrieve(state, question, settings=settings, embed_fn=embed_fn)
    if not chunks:
        return _refusal_turn(question, chunks, _REFUSAL_MESSAGE), []

    llm = llm or get_llm(settings)
    if llm.name == "none":
        return _raw_passages_turn(question, chunks)

    user_prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
        question=question,
        history=_format_history(state.messages),
        passages=_format_passages(chunks),
    )

    try:
        response = llm.complete(_SYSTEM_PROMPT, user_prompt, 0.1).strip()
    except LLMError as exc:
        # An LLM failure (no key, rate limit, network) is not the same claim
        # as "the paper doesn't contain this" - say so explicitly rather than
        # reusing the NOT_IN_PAPER wording, which would misrepresent a
        # technical failure as a grounded, content-based refusal. The chunks
        # that were retrieved are still real and useful, so surface them via
        # /sources instead of returning an empty list - see docs/decisions.md.
        logger.info("qa: LLM call failed (%s), refusing", exc)
        message = f"Could not get an answer right now: {exc.user_message}"
        return _refusal_turn(question, chunks, message), _sources_from_chunks(chunks)

    if NOT_IN_PAPER in response:
        closest_sections = ", ".join(dict.fromkeys(c.section_title for c in chunks))
        message = f"{_REFUSAL_MESSAGE} Closest sections checked: {closest_sections}."
        return _refusal_turn(question, chunks, message), _sources_from_chunks(chunks)

    turn = QATurn(
        question=question,
        answer=response,
        chunk_ids=[c.chunk_id for c in chunks],
        scores=[c.score for c in chunks],
        grounded=True,
    )
    return turn, _sources_for(response, chunks)


def qa(
    state: AgentState,
    question: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """Graph-node entry point: answers one turn and appends it to
    `state.messages`, capped to the most recent `_MAX_STORED_TURNS`.

    `question` is supplied by the CLI re-entering the graph at this node
    (Part H/I, not built yet - see docs/decisions.md) rather than by an
    upstream node, so it's an explicit argument here rather than a state
    field. The CLI's QA REPL calls `answer_question` directly instead of this
    wrapper when it needs the per-turn `sources` list for display.
    """
    turn, _sources = answer_question(
        state, question, settings=settings, llm=llm, embed_fn=embed_fn
    )
    messages = (state.messages + [turn])[-_MAX_STORED_TURNS:]
    return {"messages": messages}
