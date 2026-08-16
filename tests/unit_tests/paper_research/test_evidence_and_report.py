from open_deep_research.paper_research.evidence import extract_evidence
from open_deep_research.paper_research.models import (
    EvidenceSnippet,
    PaperAnalysis,
    PaperRecord,
    RankedPaper,
)
from open_deep_research.paper_research.report_writer import write_chinese_report


def test_extract_evidence_prefers_method_section():
    paper = PaperRecord(
        paper_id="p1",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="",
        year=2024,
        source="test",
    )
    text = """
    Abstract We study agent memory.
    Introduction Agents need memory.
    Method We introduce a long-term memory module for planning.
    Conclusion Memory improves planning.
    """

    snippets = extract_evidence(paper, text)

    assert snippets
    assert any("memory module" in snippet.excerpt for snippet in snippets)
    assert all(snippet.chinese_interpretation for snippet in snippets)


def test_write_chinese_report_contains_required_sections_and_evidence():
    paper = PaperRecord(
        paper_id="p1",
        title="Memory for LLM Agents",
        authors=["Ada"],
        abstract="A paper about memory.",
        year=2024,
        source="test",
    )
    evidence = extract_evidence(paper, "Method We introduce memory for planning.")
    analysis = PaperAnalysis(
        paper=paper,
        problem="智能体缺少长期记忆。",
        method="使用长期记忆模块。",
        contribution="提升任务规划能力。",
        limitation="实验规模有限。",
        evidence=evidence,
        evidence_quality="pdf",
    )
    ranked = [RankedPaper(paper=paper, score=10.0, reason="matched terms: memory")]

    report = write_chinese_report(
        "LLM agent memory planning",
        ["long-term memory for LLM agents"],
        ranked,
        [analysis],
    )

    for section in [
        "研究问题概述",
        "检索策略",
        "候选论文概览",
        "代表论文深读",
        "方法分类与对比",
        "主要局限",
        "研究空白",
        "后续研究方向",
        "Evidence Appendix",
    ]:
        assert section in report
    assert "Memory for LLM Agents" in report
    assert "Evidence #1" in report


def test_write_chinese_report_cites_actual_first_evidence_number_per_analysis():
    first_paper = PaperRecord(
        paper_id="p1",
        title="First Paper",
        authors=["Ada"],
        abstract="",
        year=2024,
        source="test",
        venue="ICLR",
    )
    second_paper = PaperRecord(
        paper_id="p2",
        title="Second Paper",
        authors=["Grace"],
        abstract="",
        year=2025,
        source="test",
        venue="NeurIPS",
    )
    no_evidence_paper = PaperRecord(
        paper_id="p3",
        title="No Evidence Paper",
        authors=["Lin"],
        abstract="",
        year=2026,
        source="test",
        venue="ACL",
    )
    analyses = [
        PaperAnalysis(
            paper=first_paper,
            problem="问题一。",
            method="方法一。",
            contribution="贡献一。",
            limitation="局限一。",
            evidence=[
                EvidenceSnippet(
                    paper_title=first_paper.title,
                    year=first_paper.year,
                    venue=first_paper.venue,
                    section_hint="Method",
                    excerpt="First evidence.",
                    chinese_interpretation="第一条证据。",
                )
            ],
            evidence_quality="pdf",
        ),
        PaperAnalysis(
            paper=second_paper,
            problem="问题二。",
            method="方法二。",
            contribution="贡献二。",
            limitation="局限二。",
            evidence=[
                EvidenceSnippet(
                    paper_title=second_paper.title,
                    year=second_paper.year,
                    venue=second_paper.venue,
                    section_hint="Results",
                    excerpt="Second evidence.",
                    chinese_interpretation="第二条证据。",
                )
            ],
            evidence_quality="pdf",
        ),
        PaperAnalysis(
            paper=no_evidence_paper,
            problem="问题三。",
            method="方法三。",
            contribution="贡献三。",
            limitation="局限三。",
            evidence=[],
            evidence_quality="abstract",
        ),
    ]

    report = write_chinese_report(
        "question",
        ["query"],
        [
            RankedPaper(paper=paper, score=1.0, reason="reason")
            for paper in [first_paper, second_paper, no_evidence_paper]
        ],
        analyses,
    )

    assert (
        "- First Paper: 贡献一。 [Paper: First Paper, 2024, ICLR, Evidence #1]"
        in report
    )
    assert (
        "- Second Paper: 贡献二。 [Paper: Second Paper, 2025, NeurIPS, Evidence #2]"
        in report
    )
    assert "- No Evidence Paper: 贡献三。 [Paper: No Evidence Paper, 2026, ACL]" in report


def test_write_chinese_report_renders_missing_venues_as_unknown():
    none_venue_paper = PaperRecord(
        paper_id="p1",
        title="None Venue Paper",
        authors=[],
        abstract="",
        year=2024,
        source="test",
    )
    none_venue_paper.venue = None
    blank_venue_paper = PaperRecord(
        paper_id="p2",
        title="Blank Venue Paper",
        authors=[],
        abstract="",
        year=2025,
        source="test",
        venue="",
    )
    unknown_venue_paper = PaperRecord(
        paper_id="p3",
        title="Unknown Venue Paper",
        authors=[],
        abstract="",
        year=2026,
        source="test",
        venue="unknown",
    )
    blank_snippet = EvidenceSnippet(
        paper_title=blank_venue_paper.title,
        year=blank_venue_paper.year,
        venue="",
        section_hint="Method",
        excerpt="Blank venue evidence.",
        chinese_interpretation="空 venue 证据。",
    )
    none_snippet = EvidenceSnippet(
        paper_title=none_venue_paper.title,
        year=none_venue_paper.year,
        venue="unknown",
        section_hint="Method",
        excerpt="None venue evidence.",
        chinese_interpretation="None venue 证据。",
    )
    none_snippet.venue = None
    analyses = [
        PaperAnalysis(
            paper=none_venue_paper,
            problem="问题一。",
            method="方法一。",
            contribution="贡献一。",
            limitation="局限一。",
            evidence=[none_snippet],
        ),
        PaperAnalysis(
            paper=blank_venue_paper,
            problem="问题二。",
            method="方法二。",
            contribution="贡献二。",
            limitation="局限二。",
            evidence=[blank_snippet],
        ),
    ]

    report = write_chinese_report(
        "question",
        ["query"],
        [
            RankedPaper(paper=none_venue_paper, score=1.0, reason="reason"),
            RankedPaper(paper=blank_venue_paper, score=1.0, reason="reason"),
            RankedPaper(paper=unknown_venue_paper, score=1.0, reason="reason"),
        ],
        analyses,
    )

    assert "- None Venue Paper (2024, unknown) score=1.0: reason" in report
    assert "- Blank Venue Paper (2025, unknown) score=1.0: reason" in report
    assert "- Unknown Venue Paper (2026, unknown) score=1.0: reason" in report
    assert "[Paper: None Venue Paper, 2024, unknown, Evidence #1]" in report
    assert "[Paper: Blank Venue Paper, 2025, unknown, Evidence #2]" in report
    assert "- Evidence #1: None Venue Paper (2024, unknown, Method)" in report
    assert "- Evidence #2: Blank Venue Paper (2025, unknown, Method)" in report
    assert "(2024, None)" not in report
    assert "(2025, )" not in report
    assert ", None," not in report
