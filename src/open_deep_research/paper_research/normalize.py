import re

from open_deep_research.paper_research.models import PaperRecord


def deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    for paper in papers:
        key = _dedupe_key(paper)
        if key not in merged:
            merged[key] = paper
        else:
            merged[key] = _merge_records(merged[key], paper)
    return list(merged.values())


def _dedupe_key(paper: PaperRecord) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    if paper.semantic_scholar_id:
        return f"s2:{paper.semantic_scholar_id.lower()}"
    return f"title:{_normalize_title(paper.title)}"


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _merge_records(left: PaperRecord, right: PaperRecord) -> PaperRecord:
    data = left.model_dump()
    other = right.model_dump()
    for field, value in other.items():
        if field in {"venue", "venue_source"}:
            if _is_missing(data.get(field)) and not _is_missing(value):
                data[field] = value
            continue
        if _is_missing(data.get(field)) and not _is_missing(value):
            data[field] = value
    if right.citation_count is not None:
        data["citation_count"] = max(left.citation_count or 0, right.citation_count)
    return PaperRecord(**data)


def _is_missing(value: object) -> bool:
    if isinstance(value, list):
        return value == []
    return value in {None, "", "unknown"}
