from datetime import date

import pytest

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import ArxivNotFoundError
from arxiv_agent.nodes import search_arxiv as search_arxiv_module
from arxiv_agent.nodes.search_arxiv import search_arxiv
from arxiv_agent.state import AgentState, Intent, PaperMeta


def _paper(arxiv_id: str = "2301.12345") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper",
        authors=["Ada Lovelace"],
        abstract="An abstract.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def test_paper_lookup_sets_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        search_arxiv_module.arxiv_client, "fetch_by_id", lambda _id: _paper()
    )
    state = AgentState(intent=Intent.PAPER_LOOKUP, search_query="2301.12345")

    result = search_arxiv(state, settings=Settings())

    assert result == {"selected": _paper()}


def test_paper_lookup_not_found_routes_to_rephrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_id: str):
        raise ArxivNotFoundError("nope", user_message="I couldn't find that paper.")

    monkeypatch.setattr(search_arxiv_module.arxiv_client, "fetch_by_id", _raise)
    state = AgentState(intent=Intent.PAPER_LOOKUP, search_query="9999.99999")

    result = search_arxiv(state, settings=Settings())

    assert result["next_action"] == "ask_user_to_rephrase"
    assert result["errors"] == ["I couldn't find that paper."]


def test_topic_search_stores_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    papers = [_paper("2301.00001"), _paper("2301.00002")]
    monkeypatch.setattr(
        search_arxiv_module.arxiv_client, "search", lambda query, max_results: papers
    )
    state = AgentState(intent=Intent.TOPIC_SEARCH, search_query='all:"kv cache"')

    result = search_arxiv(state, settings=Settings())

    assert result["candidates"] == papers
    assert result["needs_ranking"] is True
    assert result["warnings"] == []


def test_zero_results_broadens_query_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _search(query: str, max_results: int):
        calls.append(query)
        if len(calls) == 1:
            return []
        return [_paper()]

    monkeypatch.setattr(search_arxiv_module.arxiv_client, "search", _search)
    state = AgentState(
        intent=Intent.TOPIC_SEARCH, search_query='all:"kv cache" AND all:"compression"'
    )

    result = search_arxiv(state, settings=Settings())

    assert len(calls) == 2
    assert calls[1] == "all:kv cache"
    assert result["candidates"] == [_paper()]
    assert any("broadened" in w for w in result["warnings"])


def test_still_zero_after_broadening_asks_to_rephrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_arxiv_module.arxiv_client, "search", lambda query, max_results: []
    )
    state = AgentState(
        intent=Intent.TOPIC_SEARCH, search_query='all:"nonexistent topic"'
    )

    result = search_arxiv(state, settings=Settings())

    assert result["next_action"] == "ask_user_to_rephrase"


def test_many_results_capped_at_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    papers = [_paper(f"2301.{i:05d}") for i in range(10)]
    monkeypatch.setattr(
        search_arxiv_module.arxiv_client, "search", lambda query, max_results: papers
    )
    state = AgentState(intent=Intent.TOPIC_SEARCH, search_query="all:test")

    result = search_arxiv(state, settings=Settings(arxiv_max_results=3))

    assert len(result["candidates"]) == 3
