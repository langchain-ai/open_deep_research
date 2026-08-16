from open_deep_research.paper_research.eval import evaluate_report
from open_deep_research.paper_research.models import PaperRecord
from open_deep_research.paper_research.workflow import run_paper_research, run_paper_research_from_sources


def test_run_paper_research_generates_ranked_chinese_report_with_evidence():
    papers = [
        PaperRecord(
            paper_id=f"p{index}",
            title=f"Memory for LLM Agents {index}",
            authors=["Ada"],
            abstract="This paper studies memory and planning in autonomous agents.",
            year=2024,
            source="test",
            citation_count=10 + index,
            pdf_url=f"https://example.com/{index}.pdf",
        )
        for index in range(1, 5)
    ]
    pdf_text = {
        "p1": "Method We introduce memory for planning.",
        "p2": "Method We evaluate agent memory.",
        "p3": "Method We compare memory modules.",
    }

    report = run_paper_research("LLM agent memory planning", papers, pdf_text)

    assert len(report.planned_queries) >= 5
    assert len(report.ranked_papers) == 4
    assert len(report.analyses) == 3
    assert "研究问题概述" in report.markdown
    assert "Evidence #1" in report.markdown


def test_evaluate_report_checks_required_mvp_properties():
    papers = [
        PaperRecord(
            paper_id=f"p{index}",
            title=f"Memory Paper {index}",
            authors=["Ada"],
            abstract="This paper studies memory.",
            year=2024,
            source="test",
            pdf_url=f"https://example.com/{index}.pdf",
        )
        for index in range(1, 9)
    ]
    report = run_paper_research(
        "LLM agent memory planning",
        papers,
        {f"p{index}": "Method We introduce memory." for index in range(1, 4)},
    )

    checks = evaluate_report(report)

    assert checks["has_at_least_8_candidates"]
    assert checks["has_required_sections"]
    assert checks["has_evidence"]
    assert checks["missing_venue_marked_unknown"]


def test_run_paper_research_from_sources_retrieves_from_both_sources_without_network():
    seen_queries = []

    def arxiv_client(query):
        seen_queries.append(("arxiv", query))
        return [
            PaperRecord(
                paper_id="arxiv:2401.00001",
                title="Long-term Memory for LLM Agents",
                authors=["Ada"],
                abstract="This paper studies memory and planning in autonomous agents.",
                year=2024,
                source="arxiv",
                arxiv_id="2401.00001",
                pdf_url="https://arxiv.org/pdf/2401.00001",
            )
        ]

    def semantic_scholar_client(query):
        seen_queries.append(("semantic_scholar", query))
        return [
            PaperRecord(
                paper_id="s2:abc123",
                title="Long-term Memory for LLM Agents",
                authors=["Ada"],
                abstract="This paper studies memory and planning in autonomous agents.",
                year=2024,
                source="semantic_scholar",
                arxiv_id="2401.00001",
                semantic_scholar_id="abc123",
                venue="ICLR",
                venue_source="semantic_scholar",
                citation_count=42,
            ),
            PaperRecord(
                paper_id="s2:def456",
                title="Episodic Memory in Language Model Agents",
                authors=["Grace"],
                abstract="This paper studies episodic memory for LLM agent planning.",
                year=2023,
                source="semantic_scholar",
                semantic_scholar_id="def456",
                venue="NeurIPS",
                venue_source="semantic_scholar",
                citation_count=20,
                pdf_url="https://example.com/episodic.pdf",
            ),
        ]

    report = run_paper_research_from_sources(
        "LLM agent memory planning",
        arxiv_client=arxiv_client,
        semantic_scholar_client=semantic_scholar_client,
    )

    assert report.question == "LLM agent memory planning"
    assert len(report.planned_queries) >= 5
    assert ("arxiv", report.planned_queries[0]) in seen_queries
    assert ("semantic_scholar", report.planned_queries[0]) in seen_queries
    assert {item.paper.source for item in report.ranked_papers} == {"arxiv", "semantic_scholar"}
    assert len(report.ranked_papers) == 2
    assert report.ranked_papers[0].paper.venue in {"ICLR", "NeurIPS"}
    assert "研究问题概述" in report.markdown


def test_evaluate_report_requires_missing_venue_marker_for_missing_values():
    valid_unknown = PaperRecord(
        paper_id="p1",
        title="Memory Paper 1",
        authors=["Ada"],
        abstract="This paper studies memory.",
        year=2024,
        source="test",
        venue="unknown",
    )
    valid_none = PaperRecord.model_construct(
        paper_id="p2",
        title="Memory Paper 2",
        authors=["Ada"],
        abstract="This paper studies memory.",
        year=2024,
        source="test",
        venue=None,
    )
    invalid_empty = PaperRecord(
        paper_id="p3",
        title="Memory Paper 3",
        authors=["Ada"],
        abstract="This paper studies memory.",
        year=2024,
        source="test",
        venue="",
    )

    valid_report = run_paper_research("LLM agent memory planning", [valid_unknown, valid_none])
    invalid_report = run_paper_research("LLM agent memory planning", [valid_unknown, invalid_empty])

    assert evaluate_report(valid_report)["missing_venue_marked_unknown"]
    assert not evaluate_report(invalid_report)["missing_venue_marked_unknown"]
