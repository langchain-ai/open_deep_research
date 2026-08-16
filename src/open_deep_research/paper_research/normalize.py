import re

from open_deep_research.paper_research.models import PaperRecord


def deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    parent = list(range(len(papers)))
    key_owner: dict[str, int] = {}

    for index, paper in enumerate(papers):
        for key in _dedupe_keys(paper):
            owner = key_owner.get(key)
            if owner is None:
                key_owner[key] = index
            else:
                _union(parent, owner, index)

    merged: dict[int, PaperRecord] = {}
    for index, paper in enumerate(papers):
        root = _find(parent, index)
        if root not in merged:
            merged[root] = paper
        else:
            merged[root] = _merge_records(merged[root], paper)
    return list(merged.values())


def _dedupe_keys(paper: PaperRecord) -> list[str]:
    keys = []
    if paper.doi:
        keys.append(f"doi:{paper.doi.lower()}")
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id.lower()}")
    if paper.semantic_scholar_id:
        keys.append(f"s2:{paper.semantic_scholar_id.lower()}")
    title_key = _normalize_title(paper.title)
    if title_key:
        keys.append(f"title:{title_key}")
    return keys


def _find(parent: list[int], index: int) -> int:
    if parent[index] != index:
        parent[index] = _find(parent, parent[index])
    return parent[index]


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


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
