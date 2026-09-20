import pytest

from arxiv_agent.services.arxiv_client import extract_arxiv_id


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Modern ID format
        ("2301.12345", "2301.12345"),
        ("2301.1234", "2301.1234"),
        ("2301.12345v2", "2301.12345"),
        ("arXiv:2301.12345", "2301.12345"),
        ("please summarize 2301.12345 for me", "2301.12345"),
        # Legacy IDs
        ("cs/0701001", "cs/0701001"),
        ("math.GT/0309136", "math.gt/0309136"),
        ("hep-th/9901001v1", "hep-th/9901001"),
        # Full URLs, with/without scheme, www, export mirror, trailing .pdf
        ("https://arxiv.org/abs/2301.12345", "2301.12345"),
        ("http://www.arxiv.org/abs/2301.12345v3", "2301.12345"),
        ("arxiv.org/abs/2301.12345", "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345.pdf", "2301.12345"),
        ("https://export.arxiv.org/abs/2301.12345", "2301.12345"),
        ("https://arxiv.org/abs/math.GT/0309136", "math.gt/0309136"),
        ("arxiv.org/pdf/cs/0701001v1.pdf", "cs/0701001"),
        # Negatives: plain topics, no ID present
        ("large language models for retrieval augmented generation", None),
        ("kv cache compression techniques for long context", None),
        ("how to train a 7 billion parameter model", None),
        ("gpt-4 vs claude 3.5 benchmark comparison", None),
        ("diffusion models for image generation in 2024", None),
        ("", None),
    ],
)
def test_extract_arxiv_id(text: str, expected: str | None) -> None:
    assert extract_arxiv_id(text) == expected
