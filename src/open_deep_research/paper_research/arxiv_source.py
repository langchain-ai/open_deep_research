import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import httpx

from open_deep_research.paper_research.models import PaperRecord

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
ArxivFetcher = Callable[[str, dict[str, Any]], str]


def fetch_arxiv_papers(
    query: str,
    *,
    fetcher: ArxivFetcher | None = None,
    max_results: int = 10,
) -> list[PaperRecord]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    raw_xml = (fetcher or _httpx_fetch)(ARXIV_QUERY_URL, params)
    return parse_arxiv_feed(raw_xml)


def parse_arxiv_feed(xml: str) -> list[PaperRecord]:
    root = ET.fromstring(xml)
    papers: list[PaperRecord] = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = _clean_arxiv_id(entry.findtext(f"{ATOM}id", ""))
        title = _clean(entry.findtext(f"{ATOM}title", ""))
        abstract = _clean(entry.findtext(f"{ATOM}summary", ""))
        authors = [
            _clean(author.findtext(f"{ATOM}name", ""))
            for author in entry.findall(f"{ATOM}author")
        ]
        published = entry.findtext(f"{ATOM}published", "")
        papers.append(
            PaperRecord(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                year=_year(published),
                source="arxiv",
                arxiv_id=arxiv_id,
                venue="unknown",
                venue_source="unknown",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


def _clean(text: str) -> str:
    return " ".join(text.split())


def _clean_arxiv_id(raw_id: str) -> str:
    match = re.search(r"/abs/([^?#]+)", raw_id)
    identifier = match.group(1) if match else raw_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", identifier)


def _year(value: str) -> int | None:
    try:
        return int(value[:4])
    except ValueError:
        return None


def _httpx_fetch(url: str, params: dict[str, Any]) -> str:
    response = httpx.get(url, params=params, timeout=20.0)
    response.raise_for_status()
    return response.text
