"""Tests for graph wiring (Part H): conditional routing, the fetch_parse ->
chunk_embed straight edge, the embedding-failure -> handle_error branch, and
checkpoint persistence across separate `build_graph()` calls.

Node behavior itself is already covered by each node's own test module; these
tests use small fake nodes (via `node_overrides`) to exercise the *edges*
without paying for a real arXiv/PDF/LLM/embedding round trip.
"""

from __future__ import annotations

import sqlite3
from datetime import date

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END

from arxiv_agent.exceptions import EmbeddingError
from arxiv_agent.graph import (
    _route_after_chunk_embed,
    _route_after_search_arxiv,
    _route_after_select_paper,
    _route_after_understand_query,
    build_graph,
    build_state_graph,
    load_session,
    thread_config,
)
from arxiv_agent.state import AgentState, Intent, PaperMeta, QATurn


def _paper(arxiv_id: str = "2301.12345") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper",
        abstract="An abstract.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _memory_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


# --- routing functions -------------------------------------------------


def test_route_after_understand_query_unknown_goes_to_handle_error() -> None:
    state = AgentState(intent=Intent.UNKNOWN)
    assert _route_after_understand_query(state) == "handle_error"


def test_route_after_understand_query_known_intents_go_to_search() -> None:
    for intent in (Intent.PAPER_LOOKUP, Intent.TOPIC_SEARCH):
        assert (
            _route_after_understand_query(AgentState(intent=intent)) == "search_arxiv"
        )


def test_route_after_search_arxiv_selected_goes_to_fetch_parse() -> None:
    state = AgentState(selected=_paper())
    assert _route_after_search_arxiv(state) == "fetch_parse"


def test_route_after_search_arxiv_candidates_go_to_select_paper() -> None:
    state = AgentState(candidates=[_paper()])
    assert _route_after_search_arxiv(state) == "select_paper"


def test_route_after_search_arxiv_zero_results_ends_gracefully() -> None:
    state = AgentState(next_action="ask_user_to_rephrase")
    assert _route_after_search_arxiv(state) == END


def test_route_after_select_paper_confident_goes_to_fetch_parse() -> None:
    state = AgentState(selected=_paper(), next_action=None)
    assert _route_after_select_paper(state) == "fetch_parse"


def test_route_after_select_paper_low_confidence_ends() -> None:
    state = AgentState(next_action="confirm_with_user")
    assert _route_after_select_paper(state) == END


def test_route_after_chunk_embed_error_goes_to_handle_error() -> None:
    state = AgentState(next_action="error")
    assert _route_after_chunk_embed(state) == "handle_error"


def test_route_after_chunk_embed_ok_goes_to_summarize() -> None:
    state = AgentState()
    assert _route_after_chunk_embed(state) == "summarize"


# --- guarded chunk_embed -------------------------------------------------


def test_guarded_chunk_embed_turns_embedding_error_into_state_update() -> None:
    import arxiv_agent.graph as graph_module

    def _raise(_state: AgentState, **_kwargs) -> dict:
        raise EmbeddingError("boom", user_message="I couldn't embed the paper.")

    original = graph_module._chunk_embed
    graph_module._chunk_embed = _raise
    try:
        result = graph_module._guarded_chunk_embed(AgentState())
    finally:
        graph_module._chunk_embed = original

    assert result["next_action"] == "error"
    assert result["errors"] == ["I couldn't embed the paper."]


# --- end-to-end graph runs with fake nodes -------------------------------


def _fake_nodes(selected: PaperMeta) -> dict:
    return {
        "understand_query": lambda state: {
            "intent": Intent.PAPER_LOOKUP,
            "search_query": selected.arxiv_id,
        },
        "search_arxiv": lambda state: {"selected": selected},
        "fetch_parse": lambda state: {},
        "chunk_embed": lambda state: {"collection_name": "paper_x", "chunk_count": 1},
        "summarize": lambda state: {"briefing": None},
    }


def test_paper_lookup_happy_path_reaches_summarize() -> None:
    paper = _paper()
    graph = build_state_graph(node_overrides=_fake_nodes(paper)).compile()

    result = graph.invoke(AgentState(raw_input=paper.arxiv_id))

    assert result["selected"] == paper
    assert result["collection_name"] == "paper_x"


def test_unknown_intent_routes_through_handle_error() -> None:
    overrides = {"understand_query": lambda state: {"intent": Intent.UNKNOWN}}
    graph = build_state_graph(node_overrides=overrides).compile()

    result = graph.invoke(AgentState(raw_input=""))

    assert result["next_action"] == "error"
    assert result["errors"]


def test_low_confidence_selection_ends_without_handle_error() -> None:
    overrides = {
        "understand_query": lambda state: {
            "intent": Intent.TOPIC_SEARCH,
            "search_query": "kv cache",
        },
        "search_arxiv": lambda state: {"candidates": [_paper("1"), _paper("2")]},
        "select_paper": lambda state: {
            "candidates": state.candidates[:2],
            "next_action": "confirm_with_user",
        },
    }
    graph = build_state_graph(node_overrides=overrides).compile()

    result = graph.invoke(AgentState(raw_input="kv cache"))

    assert result["next_action"] == "confirm_with_user"
    # handle_error was never reached: no synthetic error message was added.
    assert result["errors"] == []


def test_chunk_embed_failure_routes_to_handle_error() -> None:
    import arxiv_agent.graph as graph_module

    def _raise(_state: AgentState, **_kwargs) -> dict:
        raise EmbeddingError("boom", user_message="I couldn't embed the paper.")

    overrides = _fake_nodes(_paper())
    del overrides["chunk_embed"]  # exercise the real _guarded_chunk_embed wrapper
    graph = build_state_graph(node_overrides=overrides).compile()

    original = graph_module._chunk_embed
    graph_module._chunk_embed = _raise
    try:
        result = graph.invoke(AgentState(raw_input="2301.12345"))
    finally:
        graph_module._chunk_embed = original

    assert result["next_action"] == "error"
    assert result["errors"] == ["I couldn't embed the paper."]


# --- persistence -----------------------------------------------------------


def test_state_persists_across_separate_build_graph_calls() -> None:
    paper = _paper()
    checkpointer = _memory_checkpointer()
    overrides = _fake_nodes(paper)

    graph_a = build_graph(checkpointer=checkpointer, node_overrides=overrides)
    config = thread_config(paper.arxiv_id)
    graph_a.invoke(AgentState(raw_input=paper.arxiv_id), config)

    # A second `build_graph()` sharing the same checkpointer (simulating a
    # fresh CLI process pointed at the same checkpoint DB) can resume the
    # session by thread_id without re-running anything.
    graph_b = build_graph(checkpointer=checkpointer, node_overrides=overrides)
    restored = load_session(graph_b, paper.arxiv_id)

    assert restored is not None
    assert restored.selected == paper
    assert restored.collection_name == "paper_x"


def test_load_session_returns_none_for_unknown_thread() -> None:
    checkpointer = _memory_checkpointer()
    graph = build_graph(checkpointer=checkpointer, node_overrides=_fake_nodes(_paper()))

    assert load_session(graph, "9999.99999") is None


def test_update_state_appends_qa_turn_without_a_graph_edge() -> None:
    """Simulates the CLI's QA REPL: after a session has been summarized, a
    QA turn is computed off-graph (`answer_question`) and persisted with
    `update_state` against the same thread - the mechanism `qa`'s lack of an
    incoming edge relies on (see docs/decisions.md)."""
    paper = _paper()
    checkpointer = _memory_checkpointer()
    graph = build_graph(checkpointer=checkpointer, node_overrides=_fake_nodes(paper))
    config = thread_config(paper.arxiv_id)
    graph.invoke(AgentState(raw_input=paper.arxiv_id), config)

    turn = QATurn(question="What is the method?", answer="It uses X. [1]")
    graph.update_state(config, {"messages": [turn]})

    restored = load_session(graph, paper.arxiv_id)
    assert restored is not None
    assert len(restored.messages) == 1
    assert restored.messages[0].question == "What is the method?"
