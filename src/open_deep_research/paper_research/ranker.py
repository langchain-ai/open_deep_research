import math
import re

from open_deep_research.paper_research.models import PaperRecord, RankedPaper


def rank_papers(question: str, papers: list[PaperRecord], limit: int = 10) -> list[RankedPaper]:
    ranked = [_rank_one(question, paper) for paper in papers]
    return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


def _rank_one(question: str, paper: PaperRecord) -> RankedPaper:
    question_terms = _terms(question)
    paper_text = f"{paper.title} {paper.abstract}".lower()
    matched_terms = sorted(term for term in question_terms if term in paper_text)
    relevance = 4.0 * len(matched_terms)
    recency = 2.0 if paper.year and paper.year >= 2021 else 0.0
    citations = math.log1p(paper.citation_count or 0)
    pdf = 1.5 if paper.pdf_url else 0.0
    has_venue = _has_venue_metadata(paper.venue)
    venue = 1.0 if has_venue else 0.0
    abstract_penalty = -2.0 if not paper.abstract else 0.0
    score = relevance + recency + citations + pdf + venue + abstract_penalty
    reason_parts = []
    if matched_terms:
        reason_parts.append(f"matched terms: {', '.join(matched_terms[:5])}")
    if recency:
        reason_parts.append("recent paper")
    if paper.pdf_url:
        reason_parts.append("PDF available")
    if has_venue:
        reason_parts.append(f"venue metadata: {paper.venue}")
    if paper.citation_count:
        reason_parts.append(f"{paper.citation_count} citations")
    return RankedPaper(
        paper=paper,
        score=round(score, 3),
        reason="; ".join(reason_parts) or "low metadata relevance",
    )


def _terms(text: str) -> set[str]:
    raw_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    stopwords = {"the", "and", "for", "with", "how", "does", "this", "that"}
    return {term for term in raw_terms if term not in stopwords}


def _has_venue_metadata(venue: str | None) -> bool:
    return bool(venue and venue.strip() and venue.strip().lower() != "unknown")
