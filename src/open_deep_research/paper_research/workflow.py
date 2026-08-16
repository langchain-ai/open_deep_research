from collections.abc import Callable

from open_deep_research.paper_research.arxiv_source import fetch_arxiv_papers
from open_deep_research.paper_research.evidence import extract_evidence
from open_deep_research.paper_research.models import (
    PaperAnalysis,
    PaperRecord,
    PaperResearchReport,
    RankedPaper,
)
from open_deep_research.paper_research.normalize import deduplicate_papers
from open_deep_research.paper_research.query_planner import plan_paper_queries
from open_deep_research.paper_research.ranker import rank_papers
from open_deep_research.paper_research.report_writer import write_chinese_report
from open_deep_research.paper_research.semantic_scholar_source import fetch_semantic_scholar_papers

SourceClient = Callable[[str], list[PaperRecord]]
PdfTextFetcher = Callable[[PaperRecord], str | None]


def run_paper_research(
    question: str,
    papers: list[PaperRecord],
    pdf_text_by_paper_id: dict[str, str] | None = None,
) -> PaperResearchReport:
    planned_queries = plan_paper_queries(question)
    return _build_report(question, planned_queries, papers, pdf_text_by_paper_id)


def run_paper_research_from_sources(
    question: str,
    *,
    arxiv_client: SourceClient | None = None,
    semantic_scholar_client: SourceClient | None = None,
    pdf_text_by_paper_id: dict[str, str] | None = None,
    pdf_text_fetcher: PdfTextFetcher | None = None,
) -> PaperResearchReport:
    planned_queries = plan_paper_queries(question)
    papers = _retrieve_planned_query_papers(
        planned_queries,
        arxiv_client=arxiv_client or fetch_arxiv_papers,
        semantic_scholar_client=semantic_scholar_client or fetch_semantic_scholar_papers,
    )
    return _build_report(
        question,
        planned_queries,
        papers,
        pdf_text_by_paper_id,
        pdf_text_fetcher=pdf_text_fetcher,
    )


def _retrieve_planned_query_papers(
    planned_queries: list[str],
    *,
    arxiv_client: SourceClient,
    semantic_scholar_client: SourceClient,
) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for query in planned_queries:
        papers.extend(arxiv_client(query))
        papers.extend(semantic_scholar_client(query))
    return papers


def _build_report(
    question: str,
    planned_queries: list[str],
    papers: list[PaperRecord],
    pdf_text_by_paper_id: dict[str, str] | None = None,
    *,
    pdf_text_fetcher: PdfTextFetcher | None = None,
) -> PaperResearchReport:
    unique_papers = deduplicate_papers(papers)
    ranked_papers = rank_papers(question, unique_papers, limit=10)
    pdf_text_by_paper_id = dict(pdf_text_by_paper_id or {})
    if pdf_text_fetcher:
        _fetch_pdf_text_for_top_papers(ranked_papers, pdf_text_by_paper_id, pdf_text_fetcher)
    analyses: list[PaperAnalysis] = []

    for ranked in ranked_papers:
        if len(analyses) >= 3:
            break
        paper = ranked.paper
        if not paper.pdf_url:
            continue
        text = pdf_text_by_paper_id.get(paper.paper_id, paper.abstract)
        evidence = extract_evidence(paper, text)
        analyses.append(
            PaperAnalysis(
                paper=paper,
                problem=f"该论文关注与「{question}」相关的研究问题。",
                method=evidence[0].chinese_interpretation if evidence else "方法信息主要来自摘要。",
                contribution="该论文为候选主题提供了可进一步分析的代表性证据。",
                limitation="第一版分析仍需要更细粒度的实验结果和引用上下文。",
                evidence=evidence,
                evidence_quality="pdf" if paper.paper_id in pdf_text_by_paper_id else "abstract",
            )
        )

    markdown = write_chinese_report(question, planned_queries, ranked_papers, analyses)
    return PaperResearchReport(
        question=question,
        planned_queries=planned_queries,
        ranked_papers=ranked_papers,
        analyses=analyses,
        markdown=markdown,
    )


def _fetch_pdf_text_for_top_papers(
    ranked_papers: list[RankedPaper],
    pdf_text_by_paper_id: dict[str, str],
    pdf_text_fetcher: PdfTextFetcher,
    limit: int = 3,
) -> None:
    fetched_count = 0
    for ranked in ranked_papers:
        if fetched_count >= limit:
            break
        paper = ranked.paper
        if not paper.pdf_url:
            continue
        fetched_count += 1
        if paper.paper_id in pdf_text_by_paper_id:
            continue
        text = pdf_text_fetcher(paper)
        if text:
            pdf_text_by_paper_id[paper.paper_id] = text
