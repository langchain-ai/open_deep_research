from open_deep_research.paper_research.models import PaperRecord
from open_deep_research.paper_research.normalize import deduplicate_papers
from open_deep_research.paper_research.ranker import rank_papers


def test_deduplicate_merges_arxiv_and_semantic_scholar_metadata():
    arxiv = PaperRecord(
        paper_id="arxiv:2401.00001",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="Memory for agents.",
        year=2024,
        source="arxiv",
        arxiv_id="2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
    )
    s2 = PaperRecord(
        paper_id="s2:abc",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="Memory for agents.",
        year=2024,
        source="semantic_scholar",
        arxiv_id="2401.00001",
        semantic_scholar_id="abc",
        venue="ICLR",
        venue_source="semantic_scholar",
        citation_count=100,
    )

    merged = deduplicate_papers([arxiv, s2])

    assert len(merged) == 1
    assert merged[0].venue == "ICLR"
    assert merged[0].citation_count == 100
    assert merged[0].pdf_url == "https://arxiv.org/pdf/2401.00001"


def test_deduplicate_merges_doi_record_with_arxiv_only_alias():
    doi_and_arxiv = PaperRecord(
        paper_id="doi:10.1234/memory-agents",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="Memory for agents.",
        year=2024,
        source="crossref",
        doi="10.1234/MEMORY-AGENTS",
        arxiv_id="2401.00001",
        citation_count=20,
    )
    arxiv_only = PaperRecord(
        paper_id="arxiv:2401.00001",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="Memory for agents.",
        year=2024,
        source="arxiv",
        arxiv_id="2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
        citation_count=35,
    )

    merged = deduplicate_papers([doi_and_arxiv, arxiv_only])

    assert len(merged) == 1
    assert merged[0].doi == "10.1234/MEMORY-AGENTS"
    assert merged[0].arxiv_id == "2401.00001"
    assert merged[0].pdf_url == "https://arxiv.org/pdf/2401.00001"
    assert merged[0].citation_count == 35
    assert merged[0].venue == "unknown"
    assert merged[0].venue_source == "unknown"


def test_rank_papers_prefers_relevant_recent_pdf_paper():
    relevant = PaperRecord(
        paper_id="p1",
        title="Long-term Memory for LLM Agents",
        abstract="This paper studies memory and planning in autonomous agents.",
        authors=[],
        year=2024,
        source="test",
        citation_count=50,
        pdf_url="https://example.com/p1.pdf",
    )
    irrelevant = PaperRecord(
        paper_id="p2",
        title="Database Indexing",
        abstract="This paper studies database storage.",
        authors=[],
        year=2015,
        source="test",
    )

    ranked = rank_papers("LLM agent memory planning", [irrelevant, relevant])

    assert ranked[0].paper.paper_id == "p1"
    assert ranked[0].score > ranked[1].score
    assert "memory" in ranked[0].reason.lower()
