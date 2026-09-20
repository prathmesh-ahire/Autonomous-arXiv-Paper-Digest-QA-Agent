"""Shared graph state: the data that flows between nodes and persists across
the summarization -> QA handoff."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from arxiv_agent.schemas import Briefing


class Intent(str, Enum):
    PAPER_LOOKUP = "paper_lookup"
    TOPIC_SEARCH = "topic_search"
    UNKNOWN = "unknown"


class PaperMeta(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    published: date
    updated: date | None = None
    categories: list[str] = Field(default_factory=list)
    pdf_url: str
    abs_url: str
    doi: str | None = None


class Section(BaseModel):
    title: str
    text: str
    level: int = 1
    start_page: int = 0


class ParsedPaper(BaseModel):
    full_text: str
    sections: list[Section] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    page_count: int = 0
    parse_method: str = "unknown"
    parse_quality: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class QATurn(BaseModel):
    question: str
    answer: str
    chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    grounded: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    raw_input: str = ""
    intent: Intent = Intent.UNKNOWN
    search_query: str | None = None
    candidates: list[PaperMeta] = Field(default_factory=list)
    selected: PaperMeta | None = None
    parsed: ParsedPaper | None = None
    collection_name: str | None = None
    chunk_count: int = 0
    degraded_retrieval: bool = False
    briefing: Briefing | None = None
    messages: list[QATurn] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    needs_ranking: bool = False
    selection_reason: str | None = None
    selection_scores: list[float] | None = None
    next_action: str | None = None
