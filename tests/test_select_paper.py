from datetime import date

from arxiv_agent.config import Settings
from arxiv_agent.llm.null import NullLLM
from arxiv_agent.nodes.select_paper import select_paper
from arxiv_agent.state import AgentState, PaperMeta


def _paper(arxiv_id: str, title: str = "", abstract: str = "") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=title or f"Paper {arxiv_id}",
        abstract=abstract or "An abstract.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


class _StubLLM:
    name = "stub"
    max_context_tokens = 1000

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self._response


def _embed_fn(vectors_by_text: dict[str, list[float]]):
    def embed(texts: list[str]) -> list[list[float]]:
        return [vectors_by_text[t] for t in texts]

    return embed


def test_no_llm_falls_back_to_embedding_top_pick() -> None:
    query = "query"
    best = _paper("2301.00001")
    other = _paper("2301.00002")
    vectors = {
        query: [1.0, 0.0],
        f"{best.title}\n{best.abstract}": [0.95, 0.05],
        f"{other.title}\n{other.abstract}": [0.2, 0.8],
    }
    state = AgentState(candidates=[other, best], search_query=query)

    result = select_paper(
        state,
        settings=Settings(selection_confidence_threshold=0.35),
        llm=NullLLM(),
        embed_fn=_embed_fn(vectors),
    )

    assert result["selected"] == best
    assert result["next_action"] is None


def test_llm_rerank_overrides_embedding_top_pick() -> None:
    query = "query"
    embedding_top = _paper("2301.00001")
    llm_choice = _paper("2301.00002")
    vectors = {
        query: [1.0, 0.0],
        f"{embedding_top.title}\n{embedding_top.abstract}": [0.9, 0.1],
        f"{llm_choice.title}\n{llm_choice.abstract}": [0.8, 0.2],
    }
    state = AgentState(candidates=[embedding_top, llm_choice], search_query=query)
    llm = _StubLLM('{"index": 2, "reason": "Better topical match."}')

    result = select_paper(
        state,
        settings=Settings(selection_confidence_threshold=0.35),
        llm=llm,
        embed_fn=_embed_fn(vectors),
    )

    assert result["selected"] == llm_choice
    assert result["selection_reason"] == "Better topical match."


def test_low_confidence_defers_to_user() -> None:
    query = "query"
    a, b, c = _paper("2301.00001"), _paper("2301.00002"), _paper("2301.00003")
    vectors = {
        query: [1.0, 0.0],
        f"{a.title}\n{a.abstract}": [0.1, 0.99],
        f"{b.title}\n{b.abstract}": [0.05, 0.99],
        f"{c.title}\n{c.abstract}": [0.02, 0.99],
    }
    state = AgentState(candidates=[a, b, c], search_query=query)

    result = select_paper(
        state,
        settings=Settings(selection_confidence_threshold=0.9),
        llm=NullLLM(),
        embed_fn=_embed_fn(vectors),
    )

    assert result["next_action"] == "confirm_with_user"
    assert len(result["candidates"]) == 3


def test_twenty_candidates_narrow_to_a_single_selection() -> None:
    """Failure-path coverage (Phase 40.2): a topic search returning a full
    `arxiv_max_results`-sized batch of candidates must still narrow down to
    exactly one `selected` paper, via embedding rank -> top-5 pool -> LLM
    rerank, not just error out or leave everything ambiguous."""
    query = "query"
    candidates = [_paper(f"2301.{i:05d}") for i in range(20)]
    # Give every candidate a distinct, low similarity to the query except the
    # two that should reach the LLM-rerank pool (positions 3 and 7 by score).
    vectors = {query: [1.0, 0.0]}
    for i, paper in enumerate(candidates):
        vectors[f"{paper.title}\n{paper.abstract}"] = [0.01 * i, 1.0]
    embedding_top = candidates[19]  # highest score: 0.01 * 19
    llm_choice = candidates[18]  # second-highest: within the top-5 pool
    vectors[f"{embedding_top.title}\n{embedding_top.abstract}"] = [0.5, 0.5]
    vectors[f"{llm_choice.title}\n{llm_choice.abstract}"] = [0.4, 0.6]

    state = AgentState(candidates=candidates, search_query=query)
    llm = _StubLLM('{"index": 2, "reason": "More directly on topic."}')

    result = select_paper(
        state,
        settings=Settings(selection_confidence_threshold=0.1),
        llm=llm,
        embed_fn=_embed_fn(vectors),
    )

    assert result["selected"] == llm_choice
    assert result["next_action"] is None


def test_empty_candidates_asks_to_rephrase() -> None:
    state = AgentState(candidates=[], search_query="query")

    result = select_paper(
        state,
        settings=Settings(),
        llm=NullLLM(),
        embed_fn=lambda texts: [[1.0]] * len(texts),
    )

    assert result == {"next_action": "ask_user_to_rephrase"}
