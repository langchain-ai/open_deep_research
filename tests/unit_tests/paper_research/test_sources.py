from open_deep_research.paper_research.arxiv_source import parse_arxiv_feed
from open_deep_research.paper_research.semantic_scholar_source import (
    parse_semantic_scholar_results,
)


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Memory for LLM Agents</title>
    <summary>We study memory mechanisms for agents.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <category term="cs.AI" />
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf" />
  </entry>
</feed>
"""


def test_parse_arxiv_feed_normalizes_paper():
    papers = parse_arxiv_feed(ARXIV_XML)

    assert len(papers) == 1
    assert papers[0].paper_id == "arxiv:2401.00001"
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].title == "Memory for LLM Agents"
    assert papers[0].venue == "unknown"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2401.00001"


def test_parse_semantic_scholar_results_uses_venue_and_citation_count():
    payload = {
        "data": [
            {
                "paperId": "abc123",
                "title": "Memory for LLM Agents",
                "abstract": "We study memory.",
                "year": 2024,
                "authors": [{"name": "Ada Lovelace"}],
                "venue": "ICLR",
                "citationCount": 42,
                "externalIds": {"ArXiv": "2401.00001", "DOI": "10.0000/test"},
                "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                "url": "https://semanticscholar.org/paper/abc123",
            }
        ]
    }

    papers = parse_semantic_scholar_results(payload)

    assert papers[0].paper_id == "s2:abc123"
    assert papers[0].semantic_scholar_id == "abc123"
    assert papers[0].venue == "ICLR"
    assert papers[0].venue_source == "semantic_scholar"
    assert papers[0].citation_count == 42
