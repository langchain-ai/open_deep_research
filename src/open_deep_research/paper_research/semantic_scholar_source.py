from open_deep_research.paper_research.models import PaperRecord


def parse_semantic_scholar_results(payload: dict) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for item in payload.get("data", []):
        paper_id = item.get("paperId")
        if not paper_id:
            continue
        external_ids = item.get("externalIds") or {}
        venue = item.get("venue") or "unknown"
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
