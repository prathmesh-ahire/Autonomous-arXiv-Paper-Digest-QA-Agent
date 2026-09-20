from datetime import UTC, datetime

import arxiv
import pytest

from arxiv_agent.exceptions import ArxivNotFoundError
from arxiv_agent.services.arxiv_client import fetch_by_id, search


class _FakeClient:
    """Stubs `arxiv.Client.results` so tests never touch the network."""

    def __init__(self, results: list[arxiv.Result]) -> None:
        self._results = results

    def results(self, search: arxiv.Search, offset: int = 0):
        return iter(self._results)


def _make_result(
    *,
    entry_id: str = "http://arxiv.org/abs/2301.12345v2",
    title: str = "A Great Paper\non Something",
    authors: list[str] | None = None,
    summary: str = "This paper studies\nsomething interesting.",
    doi: str = "",
    categories: list[str] | None = None,
) -> arxiv.Result:
    return arxiv.Result(
        entry_id=entry_id,
        updated=datetime(2023, 1, 2, tzinfo=UTC),
        published=datetime(2023, 1, 1, tzinfo=UTC),
        title=title,
        authors=[
            arxiv.Result.Author(name=n)
            for n in (authors if authors is not None else ["Ada Lovelace"])
        ],
        summary=summary,
        doi=doi,
        primary_category="cs.CL",
        categories=categories or ["cs.CL"],
        links=[
            arxiv.Result.Link(
                "http://arxiv.org/pdf/2301.12345v2", title="pdf", rel="related"
            )
        ],
    )


def test_fetch_by_id_maps_result_to_paper_meta() -> None:
    fake = _FakeClient([_make_result()])

    meta = fetch_by_id("2301.12345", client=fake)

    assert meta.arxiv_id == "2301.12345"
    assert meta.title == "A Great Paper on Something"
    assert meta.authors == ["Ada Lovelace"]
    assert meta.abstract == "This paper studies something interesting."
    assert meta.abs_url == "http://arxiv.org/abs/2301.12345v2"
    assert meta.pdf_url == "http://arxiv.org/pdf/2301.12345v2"
    assert meta.doi is None


def test_fetch_by_id_handles_missing_doi_and_empty_authors() -> None:
    fake = _FakeClient([_make_result(authors=[], doi="")])

    meta = fetch_by_id("2301.12345", client=fake)

    assert meta.authors == []
    assert meta.doi is None


def test_fetch_by_id_raises_when_no_results() -> None:
    fake = _FakeClient([])

    with pytest.raises(ArxivNotFoundError):
        fetch_by_id("9999.99999", client=fake)


def test_search_maps_all_results() -> None:
    fake = _FakeClient(
        [
            _make_result(entry_id="http://arxiv.org/abs/2301.00001v1"),
            _make_result(entry_id="http://arxiv.org/abs/2301.00002v1"),
        ]
    )

    results = search("kv cache compression", max_results=10, client=fake)

    assert [r.arxiv_id for r in results] == ["2301.00001", "2301.00002"]
