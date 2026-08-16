from open_deep_research.paper_research.eval import evaluate_report
from open_deep_research.paper_research.models import PaperRecord
from open_deep_research.paper_research.workflow import run_paper_research


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
