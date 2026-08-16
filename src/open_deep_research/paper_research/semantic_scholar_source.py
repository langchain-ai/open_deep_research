from collections.abc import Callable
from typing import Any

import httpx

from open_deep_research.paper_research.models import PaperRecord

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,year,authors,venue,citationCount,externalIds,openAccessPdf,url"
)
SemanticScholarFetcher = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def fetch_semantic_scholar_papers(
    query: str,
    *,
    fetcher: SemanticScholarFetcher | None = None,
    api_key: str | None = None,
    limit: int = 10,
) -> list[PaperRecord]:
    params = {
        "query": query,
        "limit": limit,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    headers = {"x-api-key": api_key} if api_key else {}
    payload = (fetcher or _httpx_fetch)(SEMANTIC_SCHOLAR_SEARCH_URL, params, headers)
    return parse_semantic_scholar_results(payload)


def parse_semantic_scholar_results(payload: dict) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for item in payload.get("data", []):
        paper_id = item.get("paperId")
        if not paper_id:
            continue
        external_ids = item.get("externalIds") or {}
        venue = _normalize_venue(item.get("venue"))
        papers.append(
            PaperRecord(
                paper_id=f"s2:{paper_id}",
                title=item.get("title") or "",
                authors=[
                    author.get("name", "")
                    for author in item.get("authors", [])
                    if author.get("name")
                ],
                abstract=item.get("abstract") or "",
                year=item.get("year"),
                source="semantic_scholar",
                arxiv_id=external_ids.get("ArXiv"),
                semantic_scholar_id=paper_id,
                doi=external_ids.get("DOI"),
                venue=venue,
                venue_source="semantic_scholar" if venue != "unknown" else "unknown",
                citation_count=item.get("citationCount"),
                pdf_url=(item.get("openAccessPdf") or {}).get("url"),
                url=item.get("url"),
            )
        )
    return papers


def _normalize_venue(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    venue = value.strip()
    return venue or "unknown"


def _httpx_fetch(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    response = httpx.get(url, params=params, headers=headers, timeout=20.0)
    response.raise_for_status()
    return response.json()
