"""Summarization node: turns `state.parsed` into a structured `Briefing`.

Builds a priority-ordered context (abstract, introduction, method, results,
conclusion) sized to the configured LLM's context window, then asks the LLM
for the briefing JSON in one pass. Papers whose selected sections are more
than 2x the available budget go through map-reduce instead: each section is
condensed to a few bullets first, and the concatenated bullets become the
context for the same briefing prompt - see `docs/decisions.md`.

Facts the LLM could get wrong (title, authors, arXiv ID, date, link) are
always overwritten from `PaperMeta` after the call, never trusted from the
model's output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from arxiv_agent.config import Settings, get_settings
from arxiv_agent.exceptions import LLMError
from arxiv_agent.llm import get_llm
from arxiv_agent.llm.base import LLMClient
from arxiv_agent.llm.json_mode import complete_json
from arxiv_agent.schemas import Briefing
from arxiv_agent.services.pdf_parser import MIN_QUALITY
from arxiv_agent.state import AgentState, ParsedPaper, Section

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarize.txt"
_SYSTEM_PROMPT = (
    "You are a meticulous research analyst who writes accurate, grounded "
    "executive briefings and never invents facts not present in the source text."
)

# Budgeting: reserve room for the prompt scaffolding (schema + worked example)
# and the model's own response, then convert the remaining tokens to a rough
# character budget for the paper text itself.
_CHARS_PER_TOKEN = 4
_RESERVED_TOKENS = 4000
_FALLBACK_CONTEXT_TOKENS = 8000  # used when the client reports no window (e.g. NullLLM)
_MIN_BUDGET_TOKENS = 1000

_MAP_REDUCE_TRIGGER_MULTIPLE = 2
_MAP_SECTION_INPUT_CHARS = 6000

_PRIORITY_SECTION_KEYWORDS: list[list[str]] = [
    ["abstract"],
    ["introduction"],
    ["method", "methods", "approach", "methodology"],
    ["result", "results", "experiment", "experiments", "evaluation"],
    ["conclusion", "discussion"],
]

_MAP_SYSTEM_PROMPT = (
    "You compress one section of a research paper into a short list of the "
    "most important, concrete facts, for later synthesis into a briefing."
)
_MAP_USER_TEMPLATE = (
    "Summarize the following paper section into 3 to 5 concise bullet points, "
    'one per line, each starting with "- ". Include only facts actually '
    "stated in the text. No other text before or after the bullets.\n\n"
    "Section: {title}\n{text}"
)


def _select_priority_sections(sections: list[Section]) -> list[Section]:
    """Pick abstract/intro/method/results/conclusion sections, in that
    priority order, skipping any that aren't present."""
    selected: list[Section] = []
    seen: set[int] = set()
    for keywords in _PRIORITY_SECTION_KEYWORDS:
        for section in sections:
            if id(section) in seen:
                continue
            title = section.title.strip().lower()
            if any(keyword in title for keyword in keywords):
                selected.append(section)
                seen.add(id(section))
                break
    return selected


def _truncate_preserving_ends(text: str, budget: int) -> str:
    """Truncate `text` to roughly `budget` chars, keeping its first and last
    paragraph intact and cutting from the middle."""
    if len(text) <= budget or budget <= 0:
        return text

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    marker = "\n...[truncated]...\n"
    if len(paragraphs) <= 2:
        half = max(budget - len(marker), 0) // 2
        return f"{text[:half]}{marker}{text[-half:]}" if half else text[:budget]

    first, last = paragraphs[0], paragraphs[-1]
    reserved = len(first) + len(last) + len(marker) + 4  # +4 for joining newlines
    remaining = max(budget - reserved, 0)
    middle = "\n\n".join(paragraphs[1:-1])[:remaining]

    if middle:
        return f"{first}\n\n{middle}{marker}{last}"
    return f"{first}{marker}{last}"


def budget_chars_for(llm: LLMClient) -> int:
    """Rough character budget for paper text, derived from the configured
    LLM's context window minus headroom for the prompt and response."""
    max_tokens = llm.max_context_tokens or _FALLBACK_CONTEXT_TOKENS
    usable_tokens = max(max_tokens - _RESERVED_TOKENS, _MIN_BUDGET_TOKENS)
    return usable_tokens * _CHARS_PER_TOKEN


def build_context(parsed: ParsedPaper, budget_chars: int) -> str:
    """Select abstract/intro/method/results/conclusion sections (falling back
    to whatever sections exist) and truncate each proportionally to fit the
    budget, preserving each section's first and last paragraph."""
    selected = _select_priority_sections(parsed.sections) or list(parsed.sections)
    if not selected:
        return parsed.full_text[:budget_chars]

    per_section_budget = max(budget_chars // len(selected), 200)
    parts = [
        f"[{section.title}]\n{_truncate_preserving_ends(section.text, per_section_budget)}"
        for section in selected
    ]
    return "\n\n".join(parts)


def _map_section(section: Section, llm: LLMClient) -> list[str]:
    text = _truncate_preserving_ends(section.text, _MAP_SECTION_INPUT_CHARS)
    user = _MAP_USER_TEMPLATE.format(title=section.title, text=text)
    try:
        response = llm.complete(_MAP_SYSTEM_PROMPT, user, 0.2)
    except LLMError as exc:
        logger.info(
            "summarize: map step failed for section %r (%s), using raw excerpt",
            section.title,
            exc,
        )
        return [text[:300].strip()]

    bullets = [
        line.lstrip("-• ").strip()
        for line in response.splitlines()
        if line.strip().startswith(("-", "•"))
    ]
    return bullets or [response.strip()[:300]]


def _map_reduce_context(sections: list[Section], llm: LLMClient) -> str:
    parts = []
    for section in sections:
        bullets = _map_section(section, llm)
        bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)
        parts.append(f"[{section.title}]\n{bullet_text}")
    return "\n\n".join(parts)


def summarize(
    state: AgentState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict:
    """Turn `state.parsed` into a structured `Briefing`, choosing single-pass
    or map-reduce summarization based on context budget."""
    settings = settings or get_settings()
    if state.selected is None:
        raise LLMError("summarize called with no selected paper")
    if state.parsed is None:
        raise LLMError("summarize called with no parsed paper")

    meta = state.selected
    parsed = state.parsed
    llm = llm or get_llm(settings)

    budget_chars = budget_chars_for(llm)
    priority_sections = _select_priority_sections(parsed.sections) or list(
        parsed.sections
    )
    raw_len = sum(len(section.text) for section in priority_sections)

    if (
        len(priority_sections) > 1
        and raw_len > budget_chars * _MAP_REDUCE_TRIGGER_MULTIPLE
    ):
        strategy = "map_reduce"
        context = _map_reduce_context(priority_sections, llm)
        logger.info(
            "summarize: %s - map-reduce over %d sections (%d raw chars vs %d budget)",
            meta.arxiv_id,
            len(priority_sections),
            raw_len,
            budget_chars,
        )
    else:
        strategy = "single_pass"
        context = build_context(parsed, budget_chars)

    user_prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
        title=meta.title,
        authors=", ".join(meta.authors) or "Unknown",
        arxiv_id=meta.arxiv_id,
        published=meta.published.isoformat(),
        link=meta.abs_url,
        context=context,
    )

    briefing = complete_json(
        llm, _SYSTEM_PROMPT, user_prompt, Briefing, temperature=0.2
    )

    # Facts the LLM could get wrong are always trusted from PaperMeta instead.
    briefing = briefing.model_copy(
        update={
            "title": meta.title,
            "authors": meta.authors,
            "arxiv_id": meta.arxiv_id,
            "published": meta.published,
            "link": meta.abs_url,
        }
    )

    notes = []
    if strategy == "map_reduce":
        notes.append(
            f"Long paper: summarized via map-reduce over {len(priority_sections)} "
            "sections before final synthesis."
        )
    if parsed.parse_quality < MIN_QUALITY or parsed.parse_method == "abstract_only":
        notes.append(
            "This briefing is based on degraded or partial paper text (low-quality "
            "PDF extraction or an abstract-only fallback); some fields may be less reliable."
        )
    if briefing.confidence_notes:
        notes.append(briefing.confidence_notes)
    if notes:
        briefing = briefing.model_copy(update={"confidence_notes": " ".join(notes)})

    return {"briefing": briefing}
