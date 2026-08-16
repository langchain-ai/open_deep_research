from open_deep_research.paper_research.evidence import extract_evidence
from open_deep_research.paper_research.models import PaperAnalysis, PaperRecord, RankedPaper
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
