from pathlib import Path

from arxiv_agent.config import Settings
from arxiv_agent.services import vectorstore
from arxiv_agent.services.chunker import Chunk


def _chunk(chunk_id: str, text: str, section: str, index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        section_title=section,
        chunk_index=index,
        char_start=0,
        char_end=len(text),
        token_estimate=len(text) // 4,
    )


def test_collection_name_sanitizes_dots_and_slashes() -> None:
    assert vectorstore.collection_name("2301.00001") == "paper_2301_00001"
    name = vectorstore.collection_name("cs/0701001")
    assert "/" not in name
    assert name[0].isalnum() and name[-1].isalnum()
    assert 3 <= len(name) <= 63


def test_collection_exists_false_when_missing(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    assert vectorstore.collection_exists("2301.00001", settings=settings) is False


def test_search_on_missing_collection_returns_empty_list(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    assert vectorstore.search("2301.00001", [1.0, 0.0], 5, settings=settings) == []


def test_upsert_then_search_roundtrip(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    arxiv_id = "2301.00002"
    chunks = [
        _chunk("c0", "[Intro] the quick brown fox", "Intro", 0),
        _chunk("c1", "[Method] an unrelated sentence about rocks", "Method", 1),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    vectorstore.upsert_chunks(arxiv_id, chunks, embeddings, settings=settings)
    results = vectorstore.search(arxiv_id, [1.0, 0.0], top_k=2, settings=settings)

    assert results[0].chunk_id == "c0"
    assert results[0].text == chunks[0].text
    assert results[0].section_title == "Intro"
    assert results[0].score > results[1].score
    assert results[0].score == 1.0  # exact match under cosine distance


def test_collection_exists_checks_chunk_count(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    arxiv_id = "2301.00003"
    chunks = [_chunk("c0", "some chunk text", "Intro", 0)]
    vectorstore.upsert_chunks(arxiv_id, chunks, [[1.0, 0.0]], settings=settings)

    assert vectorstore.collection_exists(arxiv_id, settings=settings) is True
    assert vectorstore.collection_exists(arxiv_id, 1, settings=settings) is True
    assert vectorstore.collection_exists(arxiv_id, 2, settings=settings) is False


def test_upsert_chunks_raises_on_length_mismatch(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    chunks = [_chunk("c0", "text", "Intro", 0)]
    try:
        vectorstore.upsert_chunks("2301.00004", chunks, [], settings=settings)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched lengths")


def test_upsert_chunks_empty_list_is_a_noop(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path)
    vectorstore.upsert_chunks("2301.00005", [], [], settings=settings)
    assert vectorstore.collection_exists("2301.00005", settings=settings) is False
