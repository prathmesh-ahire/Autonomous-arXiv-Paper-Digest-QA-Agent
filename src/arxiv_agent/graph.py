"""Graph assembly: wires the Part C-G nodes into one `StateGraph[AgentState]`
with conditional edges for the failure/interactive branches, and a SQLite
checkpointer so a session's state survives past a single process run.

The compiled graph's happy path runs `understand_query -> ... -> summarize ->
END`. The QA loop is *not* an edge in this graph: `qa` is registered as a node
(so it appears in the diagram and can be unit-tested in isolation) but has no
incoming edge. The CLI (Part I) re-enters an already-summarized session by
calling `nodes.qa.answer_question(...)` directly against the state loaded via
`load_session`, then persists the new turn with `compiled_graph.update_state`
against the same `thread_id` - re-entering the checkpointed thread rather than
cycling back into `qa` through a graph edge. See `docs/decisions.md`.
"""

from __future__ import annotations

import functools
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import EmbeddingError, LLMError
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.nodes.chunk_embed import chunk_embed as _chunk_embed
from arxiv_agent.nodes.fetch_parse import fetch_parse
from arxiv_agent.nodes.handle_error import handle_error
from arxiv_agent.nodes.qa import qa
from arxiv_agent.nodes.search_arxiv import search_arxiv
from arxiv_agent.nodes.select_paper import select_paper
from arxiv_agent.nodes.summarize import summarize
from arxiv_agent.nodes.understand_query import understand_query
from arxiv_agent.services.embeddings import EmbedFn
from arxiv_agent.state import AgentState, Intent

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], dict[str, Any]]

_INTERACTIVE_NEXT_ACTIONS = {"confirm_with_user", "ask_user_to_rephrase"}


def _guarded_chunk_embed(
    state: AgentState, *, settings: Settings | None = None
) -> dict:
    """Wraps `chunk_embed` so an `EmbeddingError` becomes a state update
    (`next_action = "error"`) the graph can route on, instead of aborting the
    run - LangGraph has no chance to run a conditional edge on a node that
    raised."""
    try:
        return _chunk_embed(state, settings=settings)
    except EmbeddingError as exc:
        logger.warning("graph: chunk_embed failed: %s", exc)
        return {"errors": [*state.errors, exc.user_message], "next_action": "error"}


def _safe_summarize(
    state: AgentState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict:
    """Wraps `summarize` so a missing/failing LLM (e.g. `NullLLM` when no API
    key is configured) degrades to `briefing=None` with a warning instead of
    aborting the run. `chunk_embed` already ran by this point, so parsing,
    retrieval, and QA are still usable without a briefing - matching the
    brief's "no paid API key required" constraint (see docs/decisions.md)."""
    try:
        return summarize(state, settings=settings, llm=llm)
    except LLMError as exc:
        logger.warning("graph: summarize failed: %s", exc)
        return {"warnings": [*state.warnings, exc.user_message]}


def _route_after_understand_query(state: AgentState) -> str:
    if state.intent == Intent.UNKNOWN:
        return "handle_error"
    return "search_arxiv"


def _route_after_search_arxiv(state: AgentState) -> str:
    if state.selected is not None:
        return "fetch_parse"
    if state.candidates:
        return "select_paper"
    # Zero results after the broadened retry: `search_arxiv` already set
    # `next_action = "ask_user_to_rephrase"` with the attempted queries in
    # `state.warnings` - the CLI drives the rephrase prompt from that, so the
    # graph just ends rather than routing through `handle_error` (reserved
    # for states that don't already carry their own follow-up context; see
    # docs/decisions.md).
    return END


def _route_after_select_paper(state: AgentState) -> str:
    if state.next_action in _INTERACTIVE_NEXT_ACTIONS:
        return END
    return "fetch_parse"


def _route_after_chunk_embed(state: AgentState) -> str:
    if state.next_action == "error":
        return "handle_error"
    return "summarize"


def _default_nodes(settings: Settings) -> dict[str, NodeFn]:
    """Bind the resolved `Settings` into each node as a keyword default, so a
    node invoked by the compiled graph (which calls `node_fn(state)` with no
    extra arguments) sees the same settings the CLI resolved - e.g. a
    `--provider`/`--top-k` override - instead of silently falling back to the
    process-wide `get_settings()` each node uses when called with no settings
    at all. `run_pipeline_from_selection` and the QA REPL already thread
    settings through directly since they call node functions themselves
    rather than through a compiled graph; this closes the same gap for the
    graph-driven `digest` path. See docs/decisions.md."""
    return {
        "understand_query": functools.partial(understand_query, settings=settings),
        "search_arxiv": functools.partial(search_arxiv, settings=settings),
        "select_paper": functools.partial(select_paper, settings=settings),
        "fetch_parse": functools.partial(fetch_parse, settings=settings),
        "chunk_embed": functools.partial(_guarded_chunk_embed, settings=settings),
        "summarize": functools.partial(_safe_summarize, settings=settings),
        "qa": qa,
        "handle_error": handle_error,
    }


def build_state_graph(
    *,
    settings: Settings | None = None,
    node_overrides: dict[str, NodeFn] | None = None,
) -> StateGraph:
    """Build the (uncompiled) `StateGraph`. `node_overrides` lets tests swap
    in fakes for individual nodes without touching the wiring itself."""
    nodes = _default_nodes(settings or get_settings())
    if node_overrides:
        nodes.update(node_overrides)

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("understand_query")

    graph.add_conditional_edges(
        "understand_query",
        _route_after_understand_query,
        {"search_arxiv": "search_arxiv", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_arxiv",
        _route_after_search_arxiv,
        {"fetch_parse": "fetch_parse", "select_paper": "select_paper", END: END},
    )
    graph.add_conditional_edges(
        "select_paper",
        _route_after_select_paper,
        {"fetch_parse": "fetch_parse", END: END},
    )
    # fetch_parse never crashes (it degrades to an abstract-only ParsedPaper
    # on download/parse failure) and chunk_embed already knows how to embed
    # that degraded case (a single abstract chunk, `degraded_retrieval=True`)
    # - so there is no separate "total parse failure" branch to route here;
    # see docs/decisions.md.
    graph.add_edge("fetch_parse", "chunk_embed")
    graph.add_conditional_edges(
        "chunk_embed",
        _route_after_chunk_embed,
        {"summarize": "summarize", "handle_error": "handle_error"},
    )
    graph.add_edge("summarize", END)
    graph.add_edge("handle_error", END)

    return graph


def run_pipeline_from_selection(
    state: AgentState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embed_fn: EmbedFn | None = None,
    fetch_parse_fn: Callable[..., dict] = fetch_parse,
) -> AgentState:
    """Run `fetch_parse -> chunk_embed -> summarize` directly against `state`,
    in the same order `build_state_graph` wires them.

    Used by the CLI to resume a session after the user manually confirms a
    low-confidence paper selection (`next_action == "confirm_with_user"`).
    The compiled graph only ever reaches that point by ending at `END`, not
    by pausing at an interrupt, so there is nothing for LangGraph itself to
    resume - the CLI calls the same node functions directly instead, exactly
    like the QA loop re-enters via `update_state` rather than a graph edge
    (see docs/decisions.md). `fetch_parse_fn` is overridable so the CLI's
    `--force-refresh` can inject a cache-clearing wrapper without duplicating
    this function's sequencing.
    """
    settings = settings or get_settings()
    state = state.model_copy(update=fetch_parse_fn(state, settings=settings))
    try:
        state = state.model_copy(
            update=_chunk_embed(state, settings=settings, embed_fn=embed_fn)
        )
    except EmbeddingError as exc:
        logger.warning("run_pipeline_from_selection: chunk_embed failed: %s", exc)
        return state.model_copy(
            update={"errors": [*state.errors, exc.user_message], "next_action": "error"}
        )
    state = state.model_copy(update=_safe_summarize(state, settings=settings, llm=llm))
    return state


def default_checkpointer(settings: Settings | None = None) -> SqliteSaver:
    """A `SqliteSaver` pointed at `settings.checkpoint_db`, with its schema
    already created. `check_same_thread=False` mirrors `SqliteSaver`'s own
    `from_conn_string` helper - safe here because the CLI drives the graph
    from a single thread at a time."""
    settings = settings or get_settings()
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(
    *,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    node_overrides: dict[str, NodeFn] | None = None,
) -> CompiledStateGraph:
    """Compile the graph with a SQLite checkpointer so state persists across
    process runs, keyed by `thread_id` (see `thread_config`)."""
    settings = settings or get_settings()
    graph = build_state_graph(settings=settings, node_overrides=node_overrides)
    checkpointer = checkpointer or default_checkpointer(settings)
    return graph.compile(checkpointer=checkpointer)


def thread_config(arxiv_id: str) -> dict:
    """Config scoping graph calls to one paper's session. Using the arXiv ID
    itself as `thread_id` (Phase 34.3) means re-running the same paper resumes
    its checkpointed state instead of starting a fresh one."""
    return {"configurable": {"thread_id": arxiv_id}}


def load_session(graph: CompiledStateGraph, arxiv_id: str) -> AgentState | None:
    """Restore a previously-run session's full state (including `briefing`
    and `messages`) from the checkpoint DB, or `None` if `arxiv_id` has no
    saved session yet."""
    snapshot = graph.get_state(thread_config(arxiv_id))
    if not snapshot.values:
        return None
    return AgentState.model_validate(snapshot.values)


def export_graph_diagram(
    graph: CompiledStateGraph,
    *,
    png_path: Path | str = "docs/graph.png",
    mmd_path: Path | str = "docs/graph.mmd",
) -> None:
    """Write the graph's Mermaid source to `mmd_path`, and a rendered PNG to
    `png_path` when a renderer is reachable (`draw_mermaid_png` calls out to
    mermaid.ink) - the `.mmd` text file is committed as the offline fallback
    the README embeds if the PNG render isn't available."""
    rendered = graph.get_graph()
    mermaid_source = rendered.draw_mermaid()
    Path(mmd_path).write_text(mermaid_source, encoding="utf-8")

    try:
        png_bytes = rendered.draw_mermaid_png()
    except Exception as exc:  # noqa: BLE001
        # Best-effort optional render (network call to mermaid.ink) - any
        # failure should fall back to the .mmd text file, not break the caller.
        logger.warning(  # pragma: no cover - depends on network access
            "export_graph_diagram: could not render PNG (%s); wrote %s only",
            exc,
            mmd_path,
        )
        return
    Path(png_path).write_bytes(png_bytes)
