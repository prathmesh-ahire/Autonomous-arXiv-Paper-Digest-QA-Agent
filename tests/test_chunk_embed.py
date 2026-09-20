from datetime import date
from pathlib import Path

import pytest

from arxiv_agent.config import Settings
from arxiv_agent.exceptions import EmbeddingError
from arxiv_agent.nodes.chunk_embed import chunk_embed
from arxiv_agent.services import vectorstore
from arxiv_agent.state import AgentState, PaperMeta, ParsedPaper, Section


def _meta(arxiv_id: str = "2301.00001") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title="A Paper",
        abstract="An abstract about testing chunk embedding nodes thoroughly.",
        published=date(2023, 1, 1),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _parsed(parse_method: str = "pymupdf") -> ParsedPaper:
    return ParsedPaper(
        full_text="irrelevant",
        sections=[
            Section(
                title="Introduction",
                text="A long enough introduction paragraph to survive the fifty "
                "character minimum chunk length easily.",
            ),
            Section(
                title="Method",
                text="A long enough method paragraph to also survive the fifty "
                "character minimum chunk length easily.",
            ),
        ],
        parse_method=parse_method,
    )


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    # One-hot-ish vectors, distinct enough per text for a deterministic test.
    return [[float(i == j) for j in range(len(texts))] for i in range(len(texts))]


def test_chunk_embed_stores_chunks_and_returns_handoff_fields(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    state = AgentState(selected=_meta(), parsed=_parsed())

    result = chunk_embed(state, settings=settings, embed_fn=_fake_embed_fn)

    assert result["collection_name"] == vectorstore.collection_name("2301.00001")
    assert result["chunk_count"] == 2
    assert result["degraded_retrieval"] is False
    assert vectorstore.collection_exists(
        "2301.00001", result["chunk_count"], settings=settings
    )


def test_chunk_embed_skips_reembedding_when_collection_already_matches(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path)
    state = AgentState(selected=_meta(), parsed=_parsed())
    calls = []

    def counting_embed_fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return _fake_embed_fn(texts)

    chunk_embed(state, settings=settings, embed_fn=counting_embed_fn)
    assert len(calls) == 1

    chunk_embed(state, settings=settings, embed_fn=counting_embed_fn)
    assert len(calls) == 1  # second run short-circuited, no new embed call


def test_chunk_embed_abstract_only_paper_embeds_just_the_abstract(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path)
    meta = _meta()
    parsed = ParsedPaper(
        full_text=meta.abstract,
        sections=[Section(title="Abstract", text=meta.abstract)],
        parse_method="abstract_only",
    )
    state = AgentState(selected=meta, parsed=parsed)

    result = chunk_embed(state, settings=settings, embed_fn=_fake_embed_fn)

    assert result["chunk_count"] == 1
    assert result["degraded_retrieval"] is True


def test_chunk_embed_falls_back_to_abstract_when_chunking_yields_nothing(
    tmp_path: Path,
) -> None:
    settings = Settings(chroma_dir=tmp_path)
    meta = _meta()
    parsed = ParsedPaper(
        full_text=meta.abstract,
        sections=[Section(title="Tiny", text="short")],  # < 50 chars, gets dropped
        parse_method="pymupdf",
    )
    state = AgentState(selected=meta, parsed=parsed)

    result = chunk_embed(state, settings=settings, embed_fn=_fake_embed_fn)

    assert result["chunk_count"] == 1
    assert result["degraded_retrieval"] is True


def test_chunk_embed_raises_without_parsed_paper(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    state = AgentState(selected=_meta())

    with pytest.raises(EmbeddingError):
        chunk_embed(state, settings=settings, embed_fn=_fake_embed_fn)


def test_chunk_embed_raises_without_selected_paper(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    state = AgentState(parsed=_parsed())

    with pytest.raises(EmbeddingError):
        chunk_embed(state, settings=settings, embed_fn=_fake_embed_fn)
