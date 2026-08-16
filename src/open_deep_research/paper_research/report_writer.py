from open_deep_research.paper_research.models import PaperAnalysis, RankedPaper


REQUIRED_SECTIONS = [
    "研究问题概述",
    "检索策略",
    "候选论文概览",
    "代表论文深读",
    "方法分类与对比",
    "主要局限",
    "研究空白",
    "后续研究方向",
    "Evidence Appendix",
]


def _display_venue(venue: str | None) -> str:
    if not venue or not venue.strip() or venue.strip().lower() == "unknown":
        return "unknown"
    return venue.strip()


def _analysis_evidence_numbers(analyses: list[PaperAnalysis]) -> dict[int, int]:
    first_evidence_numbers: dict[int, int] = {}
    evidence_index = 1
    for analysis_index, analysis in enumerate(analyses):
        if analysis.evidence:
            first_evidence_numbers[analysis_index] = evidence_index
            evidence_index += len(analysis.evidence)
    return first_evidence_numbers


def write_chinese_report(
    question: str,
    planned_queries: list[str],
    ranked_papers: list[RankedPaper],
    analyses: list[PaperAnalysis],
) -> str:
    first_evidence_numbers = _analysis_evidence_numbers(analyses)
    candidates = "\n".join(
        f"- {item.paper.title} ({item.paper.year or 'unknown'}, {_display_venue(item.paper.venue)}) "
        f"score={item.score}: {item.reason}"
        for item in ranked_papers
    )
    deep_reads = "\n".join(
        _format_deep_read(analysis, first_evidence_numbers.get(analysis_index))
        for analysis_index, analysis in enumerate(analyses)
    )
    methods = "\n".join(f"- {analysis.method}" for analysis in analyses)
    limitations = "\n".join(f"- {analysis.limitation}" for analysis in analyses)
    evidence_lines: list[str] = []
    evidence_index = 1
    for analysis in analyses:
        for snippet in analysis.evidence:
            evidence_lines.append(
                f"- Evidence #{evidence_index}: {snippet.paper_title} "
                f"({snippet.year or 'unknown'}, {_display_venue(snippet.venue)}, {snippet.section_hint})\n"
                f"  原文: {snippet.excerpt}\n"
                f"  解读: {snippet.chinese_interpretation}"
            )
            evidence_index += 1
    evidence = "\n".join(evidence_lines) or "- 暂无 PDF 证据，当前结论主要基于摘要。"

    return (
        f"# 计算机论文 Deep Research 报告\n\n"
        f"## 研究问题概述\n{question}\n\n"
        f"## 检索策略\n" + "\n".join(f"- {query}" for query in planned_queries) + "\n\n"
        f"## 候选论文概览\n{candidates}\n\n"
        f"## 代表论文深读\n{deep_reads}\n\n"
        f"## 方法分类与对比\n{methods}\n\n"
        f"## 主要局限\n{limitations}\n\n"
        f"## 研究空白\n当前候选论文仍需要更系统的 benchmark 对齐、长期任务评测和可复现实验比较。\n\n"
        f"## 后续研究方向\n建议继续关注带长期记忆的 agent planning、工具使用评测和多智能体协同机制。\n\n"
        f"## Evidence Appendix\n{evidence}\n"
    )


def _format_deep_read(analysis: PaperAnalysis, first_evidence_number: int | None) -> str:
    citation_parts = [
        f"Paper: {analysis.paper.title}",
        str(analysis.paper.year or "unknown"),
        _display_venue(analysis.paper.venue),
    ]
    if first_evidence_number is not None:
        citation_parts.append(f"Evidence #{first_evidence_number}")
    citation = ", ".join(citation_parts)
    return f"- {analysis.paper.title}: {analysis.contribution} [{citation}]"
