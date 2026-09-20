from datetime import date
from pathlib import Path

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm.null import NullLLM
from arxiv_agent.nodes.qa import (
    answer_question,
    qa,
    resolve_pronoun_reference,
    retrieve,
)
from arxiv_agent.services import vectorstore
from arxiv_agent.services.chunker import Chunk
from arxiv_agent.state import AgentState, PaperMeta, QATurn


def _meta(arxiv_id: str = "2301.00001") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper",
        abstract="An abstract about testing the qa node thoroughly.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _chunk(chunk_id: str, text: str, section: str, index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        section_title=section,
        chunk_index=index,
        char_start=0,
        char_end=len(text),
        token_estimate=len(text) // 4,
    )


def _fixed_embed_fn(vector: list[float]):
    calls: list[str] = []

    def fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts[0])
        return [vector]

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


class _StubLLM:
    name = "stub"
    max_context_tokens = 10_000

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls += 1
        return self._response


class _FailingLLM:
    """Simulates a live LLM failure (rate limit, network, etc.) distinct from
    an unconfigured provider - used to prove the refusal message says so
    rather than claiming the paper lacks the answer."""

    name = "failing"
    max_context_tokens = 10_000

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        raise LLMError("boom", user_message="Rate limit exceeded, try again later.")


class _ExplodingLLM:
    """Fails the test if called - used to prove the pre-LLM refusal guard
    never reaches the LLM."""

    name = "exploding"
    max_context_tokens = 10_000

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        raise AssertionError("LLM should not be called when nothing clears the floor")


# --- Phase 29: retrieval ----------------------------------------------------


def test_retrieve_drops_chunks_below_similarity_floor(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.5)
    arxiv_id = "2301.00010"
    vectorstore.upsert_chunks(
        arxiv_id,
        [
            _chunk("hi", "a highly relevant chunk of text", "Intro", 0),
            _chunk("lo", "a completely unrelated chunk of text", "Method", 1),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    results = retrieve(
        state, "question", settings=settings, embed_fn=_fixed_embed_fn([1.0, 0.0])
    )

    assert [c.chunk_id for c in results] == ["hi"]


def test_retrieve_orders_by_document_position_not_score(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.0)
    arxiv_id = "2301.00011"
    # chunk_index 1 scores higher against the query than chunk_index 0, but
    # retrieval must still return document order (0 before 1).
    vectorstore.upsert_chunks(
        arxiv_id,
        [
            _chunk("first", "earlier in the paper", "Intro", 0),
            _chunk("second", "later in the paper", "Method", 1),
        ],
        [[0.6, 0.8], [1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    results = retrieve(
        state, "question", settings=settings, embed_fn=_fixed_embed_fn([1.0, 0.0])
    )

    assert [c.chunk_id for c in results] == ["first", "second"]


def test_retrieve_dedupes_near_identical_chunks(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.0)
    arxiv_id = "2301.00012"
    shared_prefix = "x" * 150  # exceeds the dedupe prefix length
    vectorstore.upsert_chunks(
        arxiv_id,
        [
            _chunk("weaker", shared_prefix + " tail one", "Intro", 0),
            _chunk("stronger", shared_prefix + " tail two", "Intro", 1),
        ],
        [[0.8, 0.6], [1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    results = retrieve(
        state, "question", settings=settings, embed_fn=_fixed_embed_fn([1.0, 0.0])
    )

    assert len(results) == 1
    assert results[0].chunk_id == "stronger"


def test_retrieve_returns_empty_without_selected_paper(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    state = AgentState()

    results = retrieve(
        state, "question", settings=settings, embed_fn=_fixed_embed_fn([1.0, 0.0])
    )

    assert results == []


# --- Phase 30/31: grounding, refusal, conversation state --------------------


def test_answerable_question_returns_grounded_answer_with_sources(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.3)
    arxiv_id = "2301.00020"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "The method uses a transformer.", "Method", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))
    llm = _StubLLM("The paper uses a transformer architecture [1].")

    turn, sources = answer_question(
        state,
        "What architecture is used?",
        settings=settings,
        llm=llm,
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is True
    assert turn.answer == "The paper uses a transformer architecture [1]."
    assert turn.chunk_ids == ["c0"]
    assert sources == [
        {"passage": 1, "section_title": "Method", "chunk_index": 0, "score": 1.0}
    ]


def test_unanswerable_question_refuses_via_not_in_paper_token(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.3)
    arxiv_id = "2301.00021"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "The method uses a transformer.", "Method", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))
    llm = _StubLLM("NOT_IN_PAPER")

    turn, sources = answer_question(
        state,
        "What is the capital of France?",
        settings=settings,
        llm=llm,
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is False
    assert "couldn't find that" in turn.answer.lower()
    assert "Method" in turn.answer
    assert sources == [
        {"passage": 1, "section_title": "Method", "chunk_index": 0, "score": 1.0}
    ]


def test_unanswerable_question_refuses_without_calling_llm_when_nothing_clears_floor(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.9)
    arxiv_id = "2301.00022"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "an unrelated chunk", "Intro", 0)],
        [[0.0, 1.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    turn, sources = answer_question(
        state,
        "question",
        settings=settings,
        llm=_ExplodingLLM(),
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is False
    assert turn.answer == "I couldn't find that in this paper."
    assert sources == []


def test_llm_failure_refuses_with_an_honest_message_not_not_in_paper(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.3)
    arxiv_id = "2301.00026"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "The method uses a transformer.", "Method", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    turn, sources = answer_question(
        state,
        "What architecture is used?",
        settings=settings,
        llm=_FailingLLM(),
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is False
    assert "Rate limit exceeded, try again later." in turn.answer
    assert "couldn't find that in this paper" not in turn.answer.lower()
    assert sources == [
        {"passage": 1, "section_title": "Method", "chunk_index": 0, "score": 1.0}
    ]


def test_followup_with_pronoun_resolves_against_previous_question() -> None:
    messages = [
        QATurn(question="What method does the paper use?", answer="A transformer.")
    ]

    resolved = resolve_pronoun_reference("What about it?", messages)

    assert "What method does the paper use?" in resolved
    assert resolved != "What about it?"


def test_pronoun_resolution_leaves_long_questions_and_first_turns_alone() -> None:
    assert resolve_pronoun_reference("What about it?", []) == "What about it?"
    long_question = "Does the method described in this section outperform the baseline reported earlier?"
    assert (
        resolve_pronoun_reference(long_question, [QATurn(question="q", answer="a")])
        == long_question
    )


def test_retrieve_embeds_the_pronoun_resolved_question(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.0)
    arxiv_id = "2301.00023"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "some paper text", "Intro", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(
        selected=_meta(arxiv_id),
        messages=[
            QATurn(question="What method does the paper use?", answer="A transformer.")
        ],
    )
    embed_fn = _fixed_embed_fn([1.0, 0.0])

    retrieve(state, "What about it?", settings=settings, embed_fn=embed_fn)

    assert "What method does the paper use?" in embed_fn.calls[0]  # type: ignore[attr-defined]


def test_qa_node_appends_turn_and_caps_history(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.3)
    arxiv_id = "2301.00024"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "some paper text", "Intro", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))
    llm = _StubLLM("An answer [1].")

    result = qa(
        state,
        "a question",
        settings=settings,
        llm=llm,
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert len(result["messages"]) == 1
    assert result["messages"][0].question == "a question"


def test_answer_question_with_null_llm_returns_raw_passages(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.3)
    arxiv_id = "2301.00025"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "some paper text", "Intro", 0)],
        [[1.0, 0.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    turn, sources = answer_question(
        state,
        "a question",
        settings=settings,
        llm=NullLLM(),
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is True
    assert "No LLM configured" in turn.answer
    assert "some paper text" in turn.answer
    assert turn.chunk_ids == ["c0"]
    assert sources == [
        {"passage": 1, "section_title": "Intro", "chunk_index": 0, "score": 1.0}
    ]


def test_answer_question_with_null_llm_still_refuses_when_nothing_clears_floor(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path, min_similarity=0.9)
    arxiv_id = "2301.00027"
    vectorstore.upsert_chunks(
        arxiv_id,
        [_chunk("c0", "an unrelated chunk", "Intro", 0)],
        [[0.0, 1.0]],
        settings=settings,
    )
    state = AgentState(selected=_meta(arxiv_id))

    turn, sources = answer_question(
        state,
        "question",
        settings=settings,
        llm=NullLLM(),
        embed_fn=_fixed_embed_fn([1.0, 0.0]),
    )

    assert turn.grounded is False
    assert turn.answer == "I couldn't find that in this paper."
    assert sources == []
