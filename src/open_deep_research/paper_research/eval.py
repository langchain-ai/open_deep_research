from open_deep_research.paper_research.models import PaperResearchReport
from open_deep_research.paper_research.report_writer import REQUIRED_SECTIONS


def evaluate_report(report: PaperResearchReport) -> dict[str, bool]:
    return {
        "has_at_least_8_candidates": len(report.ranked_papers) >= 8,
        "top_papers_have_titles_and_abstracts": all(
            item.paper.title and item.paper.abstract for item in report.ranked_papers[:10]
        ),
        "has_at_least_3_pdf_candidates": sum(
            1 for item in report.ranked_papers if item.paper.pdf_url
        )
        >= 3,
        "has_required_sections": all(section in report.markdown for section in REQUIRED_SECTIONS),
        "has_evidence": any(analysis.evidence for analysis in report.analyses),
        "missing_venue_marked_unknown": all(
            _has_valid_venue_marker(item.paper.venue) for item in report.ranked_papers
        ),
    }


def _has_valid_venue_marker(venue: str | None) -> bool:
    if venue is None:
        return True
    normalized = venue.strip().lower()
    if normalized == "unknown":
        return True
    if normalized in {"", "n/a", "na", "none", "null", "not available"}:
        return False
    return True
