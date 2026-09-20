"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["gemini", "groq", "none"] = "none"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    llm_model: str = "gemini-3.6-flash"
    llm_requests_per_minute: int = 5

    # Paths (all default under ./data/)
    data_dir: Path = Path("./data")
    pdf_dir: Path = Path("./data/pdfs")
    chroma_dir: Path = Path("./data/chroma")
    output_dir: Path = Path("./data/outputs")
    checkpoint_db: Path = Path("./data/checkpoints/checkpoints.sqlite")

    # Tuning
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.3
    max_pdf_pages: int = 60
    max_pdf_size_mb: int = 50
    arxiv_max_results: int = 20
    selection_confidence_threshold: float = 0.35


@lru_cache
def get_settings() -> Settings:
    """Load `Settings` from the environment/.env once per process, creating
    every configured data directory."""
    settings = Settings()
    for path_field in ("data_dir", "pdf_dir", "chroma_dir", "output_dir"):
        getattr(settings, path_field).mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    return settings
