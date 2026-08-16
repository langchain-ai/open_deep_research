from open_deep_research.paper_research.models import EvidenceSnippet, PaperRecord


def test_paper_record_uses_unknown_venue_when_missing():
    paper = PaperRecord(
        paper_id="arxiv:2401.00001",
        title="Memory for LLM Agents",
        authors=["Ada Lovelace"],
        abstract="A paper about agent memory.",
        year=2024,
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2401.00001",
    )

    assert paper.venue == "unknown"
    assert paper.venue_source == "unknown"


def test_evidence_snippet_keeps_original_excerpt_and_chinese_interpretation():
    snippet = EvidenceSnippet(
        paper_title="Memory for LLM Agents",
        year=2024,
        venue="unknown",
        section_hint="Method",
        excerpt="We introduce a long-term memory module.",
        chinese_interpretation="论文提出长期记忆模块。",
    )

    assert "long-term memory" in snippet.excerpt
    assert snippet.chinese_interpretation == "论文提出长期记忆模块。"
