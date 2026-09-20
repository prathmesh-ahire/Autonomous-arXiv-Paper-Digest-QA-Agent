"""Command-line interface: `digest` runs a paper (or topic) end to end and
drops into a QA REPL; `qa` resumes an already-digested paper's REPL; `info`
and `clear-cache` are small operational helpers.

Every command is wrapped so an `AgentError` prints its `user_message` (never
a raw traceback, unless `--verbose`) and exits 1; anything unexpected exits
2. Interactive branches (`confirm_with_user`, `ask_user_to_rephrase`) are
resolved by prompting on the terminal and either resuming the pipeline
directly (`graph.run_pipeline_from_selection`) or restarting the graph with
a new topic - the compiled graph itself has no interrupt to pause at, so
"resuming" is the CLI driving node functions directly, the same pattern the
QA loop already uses (see docs/decisions.md).
"""

from __future__ import annotations

import functools
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import click
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import AgentError
from arxiv_agent.graph import (
    build_graph,
    load_session,
    run_pipeline_from_selection,
    thread_config,
)
from arxiv_agent.logging_setup import setup_logging
from arxiv_agent.nodes.fetch_parse import fetch_parse
from arxiv_agent.nodes.qa import answer_question
from arxiv_agent.services import vectorstore
from arxiv_agent.services.arxiv_client import extract_arxiv_id
from arxiv_agent.services.renderer import save_briefing, to_console
from arxiv_agent.state import AgentState, Intent, PaperMeta, QATurn

# Extracted paper text can contain arbitrary Unicode (math symbols, accented
# names, etc.). On Windows, Python's stdout/stderr default to the console's
# legacy codepage (e.g. cp1252), which can't encode most of it and crashes
# with a raw UnicodeEncodeError - reconfigure to UTF-8 before anything prints.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            pass

app = typer.Typer(
    help="Autonomous arXiv paper digest & QA agent.", no_args_is_help=True
)


def _console() -> Console:
    """A `rich.Console` that writes through Python's own (now UTF-8) stdout
    instead of rich's legacy Win32 console API, which encodes using the
    raw console codepage and can crash on non-ASCII paper text - see
    docs/decisions.md."""
    return Console(legacy_windows=False)


_PROVIDERS = ("gemini", "groq", "none")
_QA_COMMANDS = ("/exit", "/quit", "/briefing", "/sources", "/history", "/help")


# --- error handling (Phase 38.1, 38.4) --------------------------------------


def _handle_command_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        verbose = bool(kwargs.get("verbose", False))
        console = _console()
        try:
            return fn(*args, **kwargs)
        except typer.Exit:
            raise
        except AgentError as exc:
            console.print(f"[red]Error:[/red] {exc.user_message}")
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1) from None
        except (KeyboardInterrupt, click.exceptions.Abort):
            console.print("\n[dim]Cancelled.[/dim]")
            raise typer.Exit(code=1) from None
        except Exception as exc:  # noqa: BLE001 - last-resort, see module docstring
            console.print(f"[red]Unexpected internal error:[/red] {exc}")
            if verbose:
                console.print_exception()
            raise typer.Exit(code=2) from None

    return wrapper


# --- settings / display helpers ---------------------------------------------


def _resolve_settings(
    *,
    provider: str | None = None,
    output_dir: Path | None = None,
    top_k: int | None = None,
) -> Settings:
    settings = get_settings()
    if provider is not None and provider not in _PROVIDERS:
        raise typer.BadParameter(f"--provider must be one of {_PROVIDERS}")

    updates: dict[str, Any] = {}
    if provider is not None:
        updates["llm_provider"] = provider
    if output_dir is not None:
        updates["output_dir"] = output_dir
    if top_k is not None:
        updates["top_k"] = top_k
    if not updates:
        return settings

    settings = settings.model_copy(update=updates)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _print_startup_panel(console: Console, settings: Settings) -> None:
    body = (
        f"Provider: {settings.llm_provider}\n"
        f"Embedding model: {settings.embedding_model}\n"
        f"Data directory: {settings.data_dir}"
    )
    console.print(Panel(body, title="arXiv Digest Agent", border_style="cyan"))


def _clear_paper_cache(arxiv_id: str, settings: Settings) -> None:
    """Best-effort cache eviction for one paper, used by `--force-refresh`."""
    import chromadb

    (settings.pdf_dir / f"{arxiv_id}.pdf").unlink(missing_ok=True)
    try:
        vectorstore.get_client(settings=settings).delete_collection(
            vectorstore.collection_name(arxiv_id)
        )
    except chromadb.errors.NotFoundError:
        pass


def _force_refresh_fetch_parse(settings: Settings):
    def _wrapped(state: AgentState, *, settings: Settings = settings) -> dict:
        if state.selected is not None:
            _clear_paper_cache(state.selected.arxiv_id, settings)
        return fetch_parse(state, settings=settings)

    return _wrapped


# --- digest run flow (Phase 36) ---------------------------------------------


def _invoke_with_spinner(
    console: Console, graph, initial_state: AgentState, thread_id: str
) -> AgentState:
    config = thread_config(thread_id)
    with console.status("[bold cyan]starting...[/bold cyan]", spinner="dots") as status:
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name in event:
                status.update(f"[bold cyan]Running: {node_name}[/bold cyan]")
    state = load_session(graph, thread_id)
    assert state is not None  # we just invoked this thread
    return state


def _prompt_candidate_selection(
    console: Console, state: AgentState
) -> PaperMeta | None:
    candidates = state.candidates[:3]
    scores = state.selection_scores or []
    console.print(
        "[yellow]I'm not confident enough to pick automatically. Top matches:[/yellow]"
    )
    for i, paper in enumerate(candidates, start=1):
        score_text = f" (score {scores[i - 1]:.2f})" if i - 1 < len(scores) else ""
        console.print(f"  [{i}] {paper.title} - {paper.arxiv_id}{score_text}")

    choice = typer.prompt(
        "Pick a number, or press Enter to cancel", default="", show_default=False
    )
    choice = choice.strip()
    if not choice:
        return None
    try:
        index = int(choice)
    except ValueError:
        console.print("[red]Not a number - cancelling.[/red]")
        return None
    if not 1 <= index <= len(candidates):
        console.print("[red]Out of range - cancelling.[/red]")
        return None
    return candidates[index - 1]


def _run_digest(
    console: Console,
    settings: Settings,
    input_text: str,
    *,
    force_refresh: bool,
) -> AgentState | None:
    fetch_parse_fn = (
        _force_refresh_fetch_parse(settings) if force_refresh else fetch_parse
    )
    node_overrides = {"fetch_parse": fetch_parse_fn} if force_refresh else None
    graph = build_graph(settings=settings, node_overrides=node_overrides)

    raw_input = input_text
    while True:
        # A fresh thread per attempt (not one shared across rephrase retries):
        # `search_arxiv`'s success path doesn't reset `next_action`, so a
        # retry re-invoked on the *same* thread would inherit the previous
        # attempt's stale `next_action="ask_user_to_rephrase"` from the
        # checkpoint and loop forever even after a successful search - see
        # docs/decisions.md. Recognizable arXiv IDs still get their own
        # canonical thread either way, so an ID typed as a "different topic"
        # still resumes that paper's session correctly.
        arxiv_id = extract_arxiv_id(raw_input)
        thread_id = arxiv_id or f"session-{uuid.uuid4().hex[:12]}"
        if arxiv_id and force_refresh:
            _clear_paper_cache(arxiv_id, settings)

        state = _invoke_with_spinner(
            console, graph, AgentState(raw_input=raw_input), thread_id
        )

        if state.next_action == "ask_user_to_rephrase":
            if state.warnings:
                console.print(f"[yellow]{state.warnings[-1]}[/yellow]")
            console.print("[yellow]No matching papers found.[/yellow]")
            new_topic = typer.prompt(
                "Try a different topic (or press Enter to cancel)",
                default="",
                show_default=False,
            )
            if not new_topic.strip():
                return None
            raw_input = new_topic
            continue

        if state.next_action == "confirm_with_user":
            chosen = _prompt_candidate_selection(console, state)
            if chosen is None:
                return None
            state = state.model_copy(update={"selected": chosen, "next_action": None})
            with console.status(
                "[bold cyan]fetch_parse -> chunk_embed -> summarize[/bold cyan]"
            ):
                state = run_pipeline_from_selection(
                    state, settings=settings, fetch_parse_fn=fetch_parse_fn
                )

        break

    if state.selected is not None and thread_id != state.selected.arxiv_id:
        graph.update_state(thread_config(state.selected.arxiv_id), dict(state))
    return state


@app.command()
@_handle_command_errors
def digest(
    input_text: str = typer.Argument(
        ..., help="An arXiv ID, an arxiv.org URL, or a topic to search for."
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", help="QA retrieval depth (defaults to settings.top_k)."
    ),
    no_qa: bool = typer.Option(
        False, "--no-qa", help="Skip the interactive QA REPL after the briefing."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Where to save the briefing."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM_PROVIDER: gemini|groq|none."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Debug logs and full tracebacks on error."
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Re-download the PDF and re-embed even if cached.",
    ),
) -> None:
    """Fetch, parse, and summarize a paper into a structured briefing, then
    open a grounded QA session over it."""
    setup_logging("DEBUG" if verbose else "INFO")
    settings = _resolve_settings(provider=provider, output_dir=output_dir, top_k=top_k)
    console = _console()
    _print_startup_panel(console, settings)

    state = _run_digest(console, settings, input_text, force_refresh=force_refresh)
    if state is None:
        console.print("Cancelled.")
        raise typer.Exit(code=0)

    if state.next_action == "error":
        message = (
            state.errors[-1]
            if state.errors
            else "The agent could not complete this request."
        )
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(code=1 if state.intent == Intent.UNKNOWN else 2)

    if state.briefing is None and state.selected is None:
        console.print("[red]No briefing was produced.[/red]")
        raise typer.Exit(code=2)

    if state.briefing is not None:
        to_console(state.briefing, console=console)
        md_path, json_path = save_briefing(state.briefing, settings.output_dir)
        console.print(f"[dim]Saved: {md_path}  {json_path}[/dim]")
    else:
        # Parsing/retrieval succeeded but summarization couldn't run (e.g. no
        # LLM API key configured) - degrade to QA-only instead of failing the
        # whole run, per the brief's "no paid API key required" constraint.
        console.print(
            "[yellow]No briefing was generated (see the note below), but the "
            "paper was parsed and indexed - you can still ask questions.[/yellow]"
        )
    for warning in state.warnings:
        console.print(f"[yellow]Note:[/yellow] {warning}")

    if not no_qa:
        _qa_repl(console, settings, state)


# --- QA REPL (Phase 37) -----------------------------------------------------


def _print_sources(console: Console, sources: list[dict]) -> None:
    if not sources:
        console.print("[dim](no cited sources for the last answer)[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Passage")
    table.add_column("Section")
    table.add_column("Chunk")
    table.add_column("Score")
    for source in sources:
        table.add_row(
            f"[{source['passage']}]",
            source["section_title"],
            str(source["chunk_index"]),
            f"{source['score']:.2f}",
        )
    console.print(table)


def _print_history(console: Console, messages: list[QATurn]) -> None:
    if not messages:
        console.print("[dim](no questions asked yet)[/dim]")
        return
    for turn in messages:
        label = "grounded" if turn.grounded else "refused"
        console.print(f"[bold]Q:[/bold] {turn.question}  [dim]({label})[/dim]")
        console.print(f"[bold]A:[/bold] {turn.answer}\n")


def _print_answer(console: Console, turn: QATurn, sources: list[dict]) -> None:
    if turn.grounded:
        body = turn.answer
        if sources:
            citations = "\n".join(
                f"[{s['passage']}] {s['section_title']} (chunk {s['chunk_index']}, score {s['score']:.2f})"
                for s in sources
            )
            body = f"{body}\n\n{citations}"
        console.print(Panel(body, title="Answer", border_style="green"))
    else:
        console.print(
            Panel(turn.answer, title="Not Found in Paper", border_style="red")
        )


def _qa_repl(
    console: Console, settings: Settings, state: AgentState, graph=None
) -> None:
    if state.selected is None:
        return
    if graph is None:
        graph = build_graph(settings=settings)
    arxiv_id = state.selected.arxiv_id
    config = thread_config(arxiv_id)

    console.print(
        Panel(
            "Ask a question about the paper. Commands: " + "  ".join(_QA_COMMANDS),
            title="QA",
            border_style="cyan",
        )
    )
    last_sources: list[dict] = []

    while True:
        try:
            question = typer.prompt(">", prompt_suffix=" ").strip()
        except (EOFError, KeyboardInterrupt, click.exceptions.Abort):
            console.print("\n[dim]Ending session - progress is already saved.[/dim]")
            return

        if not question:
            continue
        if question in ("/exit", "/quit"):
            return
        if question == "/help":
            console.print("Commands: " + "  ".join(_QA_COMMANDS))
            continue
        if question == "/briefing":
            if state.briefing is not None:
                to_console(state.briefing, console=console)
            else:
                console.print("[dim](no briefing available)[/dim]")
            continue
        if question == "/sources":
            _print_sources(console, last_sources)
            continue
        if question == "/history":
            _print_history(console, state.messages)
            continue

        turn, sources = answer_question(state, question, settings=settings)
        state = state.model_copy(update={"messages": (state.messages + [turn])[-20:]})
        last_sources = sources
        graph.update_state(config, {"messages": state.messages})
        _print_answer(console, turn, sources)


@app.command()
@_handle_command_errors
def qa(
    arxiv_id: str = typer.Argument(
        ..., help="The arXiv ID of a previously-digested paper."
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", help="QA retrieval depth (defaults to settings.top_k)."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM_PROVIDER: gemini|groq|none."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Debug logs and full tracebacks on error."
    ),
) -> None:
    """Resume the QA REPL for a paper that was already digested."""
    setup_logging("DEBUG" if verbose else "INFO")
    settings = _resolve_settings(provider=provider, top_k=top_k)
    console = _console()

    normalized = extract_arxiv_id(arxiv_id) or arxiv_id
    graph = build_graph(settings=settings)
    state = load_session(graph, normalized)
    if state is None or state.selected is None:
        console.print(
            f"[red]No saved session found for '{normalized}'.[/red] "
            f"Run [bold]digest {normalized}[/bold] first."
        )
        raise typer.Exit(code=1)

    if state.briefing is not None:
        to_console(state.briefing, console=console)
    _qa_repl(console, settings, state, graph=graph)


# --- utility commands (Phase 38.2, 38.3) ------------------------------------


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


@app.command()
@_handle_command_errors
def info(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Show resolved configuration, cache locations, and disk usage."""
    settings = get_settings()
    console = _console()

    table = Table(
        title="arXiv Digest Agent - Configuration",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("LLM provider", settings.llm_provider)
    table.add_row("LLM model", settings.llm_model)
    table.add_row("Gemini API key set", "yes" if settings.gemini_api_key else "no")
    table.add_row("Groq API key set", "yes" if settings.groq_api_key else "no")
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row(
        "Chunk size / overlap", f"{settings.chunk_size} / {settings.chunk_overlap}"
    )
    table.add_row(
        "Top-k / min similarity", f"{settings.top_k} / {settings.min_similarity}"
    )
    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Output directory", str(settings.output_dir))
    table.add_row(
        "PDF cache", f"{settings.pdf_dir} ({_dir_size_mb(settings.pdf_dir):.1f} MB)"
    )
    table.add_row(
        "Chroma vector store",
        f"{settings.chroma_dir} ({_dir_size_mb(settings.chroma_dir):.1f} MB)",
    )
    table.add_row(
        "LLM response cache",
        f"{settings.data_dir / 'llm_cache'} ({_dir_size_mb(settings.data_dir / 'llm_cache'):.1f} MB)",
    )
    table.add_row(
        "Checkpoint DB",
        f"{settings.checkpoint_db} ({_dir_size_mb(settings.checkpoint_db):.1f} MB)",
    )
    console.print(table)


@app.command(name="clear-cache")
@_handle_command_errors
def clear_cache(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Delete cached PDFs, embeddings, checkpoints, and LLM responses."""
    settings = get_settings()
    console = _console()

    targets = [
        settings.pdf_dir,
        settings.chroma_dir,
        settings.data_dir / "llm_cache",
        settings.checkpoint_db,
    ]
    console.print("This will delete:")
    for target in targets:
        console.print(f"  - {target}")

    if not yes and not typer.confirm("Continue?"):
        console.print("Cancelled.")
        raise typer.Exit(code=0)

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            target.unlink()
    console.print("[green]Cache cleared.[/green]")


if __name__ == "__main__":
    app()
