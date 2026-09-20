from datetime import date

import pytest

from arxiv_agent.exceptions import LLMError
from arxiv_agent.services.ranker import embed_rank, llm_rank
from arxiv_agent.state import PaperMeta


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


def _fake_embed_fn(vectors_by_text: dict[str, list[float]]):
    def embed(texts: list[str]) -> list[list[float]]:
        return [vectors_by_text[t] for t in texts]

    return embed


def test_embed_rank_orders_by_cosine_similarity() -> None:
    query = "query"
    close = _paper("2301.00001")
    far = _paper("2301.00002")
    vectors = {
        query: [1.0, 0.0],
        f"{close.title}\n{close.abstract}": [0.9, 0.1],
        f"{far.title}\n{far.abstract}": [0.1, 0.9],
    }

    ranked = embed_rank(query, [far, close], embed_fn=_fake_embed_fn(vectors))

    assert [paper.arxiv_id for paper, _ in ranked] == ["2301.00001", "2301.00002"]
    assert ranked[0][1] > ranked[1][1]


def test_embed_rank_empty_candidates_returns_empty_list() -> None:
    assert embed_rank("query", [], embed_fn=lambda texts: [[1.0]] * len(texts)) == []


def test_llm_rank_returns_zero_based_index_and_reason() -> None:
    papers = [_paper("2301.00001"), _paper("2301.00002")]
    llm = _StubLLM('{"index": 2, "reason": "It directly addresses the query."}')

    index, reason = llm_rank("query", papers, llm)

    assert index == 1
    assert reason == "It directly addresses the query."


def test_llm_rank_raises_on_out_of_range_index() -> None:
    papers = [_paper("2301.00001")]
    llm = _StubLLM('{"index": 5, "reason": "bogus"}')

    with pytest.raises(LLMError):
        llm_rank("query", papers, llm)
