"""arXiv ID parsing and a thin wrapper around the `arxiv` package.

`extract_arxiv_id` recognizes the modern `NNNN.NNNNN[vN]` format, legacy
`archive[.CLASS]/NNNNNNN[vN]` IDs, and arxiv.org `abs`/`pdf` URLs (with or
without scheme, `www`, or the `export.` mirror prefix).
"""

from __future__ import annotations

import re
from functools import lru_cache

import arxiv

from arxiv_agent.exceptions import ArxivNotFoundError
from arxiv_agent.state import PaperMeta

# Boundary lookarounds keep these from matching inside longer numbers/words
# (e.g. a plain decimal like "3.14159" or a version string) while still
# picking the ID out of "arXiv:2301.12345" or a sentence.
_NEW_ID_RE = re.compile(r"(?<![\w.])\d{4}\.\d{4,5}(?:v\d+)?(?![\w.])")
_LEGACY_ID_RE = re.compile(
    r"(?<![\w./])[a-zA-Z-]+(?:\.[a-zA-Z]{2,3})?/\d{7}(?:v\d+)?(?![\w])"
)
_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:export\.)?arxiv\.org/(?:abs|pdf)/([^\s?#]+)",
    re.IGNORECASE,
)


def extract_arxiv_id(text: str) -> str | None:
    """Pull a normalized arXiv ID out of a bare ID, a URL, or free text.

    Returns `None` when no ID-shaped substring is found (e.g. a plain topic).
    """
    text = text.strip()
    if not text:
        return None

    url_match = _URL_RE.search(text)
    if url_match:
        candidate = url_match.group(1)
        if candidate.lower().endswith(".pdf"):
            candidate = candidate[: -len(".pdf")]
        return _normalize(candidate)

    legacy_match = _LEGACY_ID_RE.search(text)
    if legacy_match:
        return _normalize(legacy_match.group(0))

    new_match = _NEW_ID_RE.search(text)
    if new_match:
        return _normalize(new_match.group(0))

    return None


def _normalize(candidate: str) -> str:
    """Strip a trailing version suffix and lowercase the category prefix."""
    candidate = candidate.strip().strip("/")
    candidate = re.sub(r"v\d+$", "", candidate)
    if "/" in candidate:
        prefix, _, rest = candidate.partition("/")
        candidate = f"{prefix.lower()}/{rest}"
    return candidate


@lru_cache
def _default_client() -> arxiv.Client:
    # delay_seconds=3 / num_retries=3 follow arXiv's requested politeness
    # policy for the API (see docs/decisions.md).
    return arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)


def fetch_by_id(arxiv_id: str, *, client: arxiv.Client | None = None) -> PaperMeta:
    """Fetch a single paper's metadata by its arXiv ID."""
    client = client or _default_client()
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        result = next(client.results(search))
    except StopIteration:
        raise ArxivNotFoundError(
            f"No arXiv paper found for ID {arxiv_id!r}",
            user_message=(
                f"I couldn't find an arXiv paper with ID '{arxiv_id}'. "
                "Check the ID and try again."
            ),
        ) from None
    return _to_meta(result)


def search(
    query: str,
    max_results: int = 20,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    *,
    client: arxiv.Client | None = None,
) -> list[PaperMeta]:
    """Search arXiv for papers matching `query`, most relevant first."""
    client = client or _default_client()
    arxiv_search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
    return [_to_meta(result) for result in client.results(arxiv_search)]


def _to_meta(result: arxiv.Result) -> PaperMeta:
    """Map an `arxiv.Result` onto our `PaperMeta`, tolerating missing DOI/authors."""
    normalized_id = re.sub(r"v\d+$", "", result.get_short_id())
    return PaperMeta(
        arxiv_id=normalized_id,
        title=" ".join(result.title.split()),
        authors=[author.name for author in result.authors],
        abstract=" ".join(result.summary.split()),
        published=result.published.date(),
        updated=result.updated.date() if result.updated else None,
        categories=list(result.categories),
        pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{normalized_id}",
        abs_url=result.entry_id,
        doi=result.doi or None,
    )
