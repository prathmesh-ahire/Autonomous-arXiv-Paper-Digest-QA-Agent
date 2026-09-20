from datetime import date
from pathlib import Path

import pytest
import requests

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import PDFDownloadError
from arxiv_agent.services.pdf_parser import download_pdf
from arxiv_agent.state import PaperMeta

_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def _paper(arxiv_id: str = "2301.12345") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper",
        abstract="An abstract.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content
        self.closed = False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self) -> None:
        self.closed = True


def test_download_pdf_saves_valid_pdf(tmp_path: Path) -> None:
    content = _FIXTURE_PDF.read_bytes()
    calls: list[str] = []

    def get_fn(url: str, **kwargs):
        calls.append(url)
        return _FakeResponse(content)

    meta = _paper()
    result = download_pdf(meta, tmp_path, settings=Settings(), get_fn=get_fn)

    assert result == tmp_path / "2301.12345.pdf"
    assert result.read_bytes() == content
    assert calls == [meta.pdf_url]


def test_download_pdf_cache_hit_skips_network(tmp_path: Path) -> None:
    meta = _paper()
    dest = tmp_path / "2301.12345.pdf"
    dest.write_bytes(b"%PDF-1.4 already here")

    def get_fn(url: str, **kwargs):
        raise AssertionError("should not hit the network on a cache hit")

    result = download_pdf(meta, tmp_path, settings=Settings(), get_fn=get_fn)

    assert result == dest
    assert result.read_bytes() == b"%PDF-1.4 already here"


def test_download_pdf_legacy_id_sanitizes_filename(tmp_path: Path) -> None:
    content = _FIXTURE_PDF.read_bytes()
    meta = _paper("cs/0701001")

    result = download_pdf(
        meta,
        tmp_path,
        settings=Settings(),
        get_fn=lambda url, **kw: _FakeResponse(content),
    )

    assert result.name == "cs_0701001.pdf"


def test_download_pdf_rejects_non_pdf_content(tmp_path: Path) -> None:
    meta = _paper()

    with pytest.raises(PDFDownloadError):
        download_pdf(
            meta,
            tmp_path,
            settings=Settings(),
            get_fn=lambda url, **kw: _FakeResponse(b"<html>not a pdf</html>"),
        )

    assert list(tmp_path.iterdir()) == []


def test_download_pdf_enforces_max_size(tmp_path: Path) -> None:
    content = _FIXTURE_PDF.read_bytes()
    assert len(content) > 0
    meta = _paper()
    settings = Settings(max_pdf_size_mb=0)

    with pytest.raises(PDFDownloadError, match="exceeds max size"):
        download_pdf(
            meta,
            tmp_path,
            settings=settings,
            get_fn=lambda url, **kw: _FakeResponse(content),
        )

    assert list(tmp_path.iterdir()) == []


def test_download_pdf_wraps_network_errors(tmp_path: Path) -> None:
    meta = _paper()

    def get_fn(url: str, **kwargs):
        raise requests.ConnectionError("boom")

    with pytest.raises(PDFDownloadError):
        download_pdf(meta, tmp_path, settings=Settings(), get_fn=get_fn)
