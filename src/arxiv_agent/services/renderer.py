"""Renders a `Briefing` to Markdown, JSON, and the terminal (rich)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arxiv_agent.schemas import Briefing


def to_markdown(briefing: Briefing) -> str:
    """Render a briefing as Markdown with headings matching the assessment's
    required field list."""
    lines = [
        f"# {briefing.title}",
        "",
        f"**Authors:** {', '.join(briefing.authors) or 'Unknown'}  ",
        f"**arXiv ID:** {briefing.arxiv_id}  ",
        f"**Published:** {briefing.published.isoformat()}  ",
        f"**Link:** {briefing.link}",
        "",
    ]
    if briefing.confidence_notes:
        lines += [f"> **Note:** {briefing.confidence_notes}", ""]

    lines += [
        "## Why It Matters",
        "",
        briefing.why_it_matters,
        "",
        "## Problem Statement",
        "",
        briefing.problem_statement,
        "",
        "## Method",
        "",
        *[f"- {item}" for item in briefing.method],
        "",
        "## Key Results",
        "",
        *[f"- {item}" for item in briefing.key_results],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in briefing.limitations],
        "",
        "## Suggested Follow-up Questions",
        "",
        *[f"- {item}" for item in briefing.followup_questions],
        "",
    ]
    return "\n".join(lines)


def to_json(briefing: Briefing) -> str:
    """Render a briefing as indented JSON."""
    return briefing.model_dump_json(indent=2)


def to_console(briefing: Briefing, *, console: Console | None = None) -> None:
    """Render a briefing to the terminal as rich panels and a details table."""
    # legacy_windows=False avoids rich's raw Win32 console API, which encodes
    # using the console's codepage (e.g. cp1252) and can crash on non-ASCII
    # paper text (accented author names, math symbols) - see docs/decisions.md.
    console = console or Console(legacy_windows=False)

    header = (
        f"[bold]{briefing.title}[/bold]\n"
        f"Authors: {', '.join(briefing.authors) or 'Unknown'}\n"
        f"arXiv: {briefing.arxiv_id}  |  Published: {briefing.published.isoformat()}\n"
        f"Link: {briefing.link}"
    )
    console.print(Panel(header, title="Executive Briefing", border_style="cyan"))

    if briefing.confidence_notes:
        console.print(
            Panel(
                briefing.confidence_notes,
                title="Confidence Notes",
                border_style="yellow",
            )
        )

    console.print(Panel(briefing.why_it_matters, title="Why It Matters"))
    console.print(Panel(briefing.problem_statement, title="Problem Statement"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Section")
    table.add_column("Details")
    table.add_row("Method", "\n".join(f"• {item}" for item in briefing.method))
    table.add_row(
        "Key Results", "\n".join(f"• {item}" for item in briefing.key_results)
    )
    table.add_row(
        "Limitations", "\n".join(f"• {item}" for item in briefing.limitations)
    )
    table.add_row(
        "Follow-up Questions",
        "\n".join(f"• {item}" for item in briefing.followup_questions),
    )
    console.print(table)


def save_briefing(briefing: Briefing, output_dir: Path) -> tuple[Path, Path]:
    """Write `{arxiv_id}.md` and `{arxiv_id}.json` into `output_dir`, creating
    it if needed. Returns `(markdown_path, json_path)`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{briefing.arxiv_id}.md"
    json_path = output_dir / f"{briefing.arxiv_id}.json"
    md_path.write_text(to_markdown(briefing), encoding="utf-8")
    json_path.write_text(to_json(briefing), encoding="utf-8")
    return md_path, json_path
