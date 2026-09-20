"""Exercises the real sentence-transformers model (unlike other services'
tests, which inject a fake `embed_fn` - see docs/decisions.md): this module
*is* the embedding wrapper, so a fake would test nothing. `get_embedder` is
process-cached, so this only pays the model-load cost once per test run.
"""

import pytest

from arxiv_agent.services.embeddings import embed_query, embed_texts

pytestmark = pytest.mark.network


def test_embed_texts_returns_normalized_vectors_of_the_expected_dimension() -> None:
    vectors = embed_texts(["hello world", "a different sentence entirely"])

    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 384
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-4


def test_embed_texts_identical_inputs_give_identical_vectors() -> None:
    vectors = embed_texts(["the same text", "the same text"])

    assert vectors[0] == vectors[1]


def test_embed_texts_empty_list_returns_empty_list() -> None:
    assert embed_texts([]) == []


def test_embed_query_matches_embed_texts_single_call() -> None:
    text = "a query about transformer attention"

    assert embed_query(text) == embed_texts([text])[0]
