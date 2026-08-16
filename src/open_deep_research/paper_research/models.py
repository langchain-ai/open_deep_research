from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    source: str
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    doi: str | None = None
    venue: str = "unknown"
    venue_source: str = "unknown"
    citation_count: int | None = None
    pdf_url: str | None = None
    url: str | None = None


class RankedPaper(BaseModel):
    paper: PaperRecord
    score: float
    reason: str


class EvidenceSnippet(BaseModel):
    paper_title: str
    year: int | None = None
    venue: str = "unknown"
    section_hint: str = "unknown"
    excerpt: str
    chinese_interpretation: str


class PaperAnalysis(BaseModel):
    paper: PaperRecord
    problem: str
    method: str
    contribution: str
    limitation: str
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    evidence_quality: str = "abstract"


class PaperResearchReport(BaseModel):
    question: str
    planned_queries: list[str]
    ranked_papers: list[RankedPaper]
    analyses: list[PaperAnalysis]
    markdown: str
