import re

from open_deep_research.paper_research.models import EvidenceSnippet, PaperRecord


def extract_evidence(
    paper: PaperRecord,
    text: str,
    max_snippets: int = 3,
) -> list[EvidenceSnippet]:
    chunks = _candidate_chunks(text)
    snippets: list[EvidenceSnippet] = []
    for index, chunk in enumerate(chunks[:max_snippets], start=1):
        snippets.append(
            EvidenceSnippet(
                paper_title=paper.title,
                year=paper.year,
                venue=paper.venue,
                section_hint=_section_hint(chunk),
                excerpt=_excerpt(chunk),
                chinese_interpretation=f"证据 {index} 表明：该论文讨论了与研究问题相关的方法或发现。",
            )
        )
    return snippets


def _candidate_chunks(text: str) -> list[str]:
    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    priority_terms = ["method", "approach", "propose", "introduce", "experiment", "result", "limitation"]
    prioritized = [
        sentence
        for sentence in sentences
        if any(term in sentence.lower() for term in priority_terms)
    ]
    return prioritized or sentences[:3]


def _section_hint(chunk: str) -> str:
    lower = chunk.lower()
    if "method" in lower or "approach" in lower:
        return "Method"
    if "experiment" in lower or "result" in lower:
        return "Experiment"
    if "limitation" in lower or "future" in lower:
        return "Limitation"
    return "unknown"


def _excerpt(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."
