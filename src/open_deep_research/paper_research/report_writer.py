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


def write_chinese_report(
    question: str,
    planned_queries: list[str],
    ranked_papers: list[RankedPaper],
    analyses: list[PaperAnalysis],
) -> str:
    candidates = "\n".join(
        f"- {item.paper.title} ({item.paper.year or 'unknown'}, {item.paper.venue}) "
        f"score={item.score}: {item.reason}"
        for item in ranked_papers
    )
    deep_reads = "\n".join(
        f"- {analysis.paper.title}: {analysis.contribution} "
        f"[Paper: {analysis.paper.title}, {analysis.paper.year or 'unknown'}, "
        f"{analysis.paper.venue}, Evidence #1]"
        for analysis in analyses
    )
    methods = "\n".join(f"- {analysis.method}" for analysis in analyses)
    limitations = "\n".join(f"- {analysis.limitation}" for analysis in analyses)
    evidence_lines: list[str] = []
    evidence_index = 1
    for analysis in analyses:
        for snippet in analysis.evidence:
            evidence_lines.append(
                f"- Evidence #{evidence_index}: {snippet.paper_title} "
                f"({snippet.year or 'unknown'}, {snippet.venue}, {snippet.section_hint})\n"
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
