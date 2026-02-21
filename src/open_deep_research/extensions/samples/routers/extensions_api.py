"""Extensions API router - data analysis pipeline and combined research+analysis.

Endpoints:
    POST /api/extensions/analyze          - Run analysis pipeline on provided data
    POST /api/extensions/research-analyze - Core research → analysis pipeline → HTML report
    GET  /api/extensions/charts/{filename}  - Serve chart HTML files
    GET  /api/extensions/reports/{filename} - Serve HTML report files
    GET  /api/extensions/download/{report_id}/research - Download research markdown
    GET  /api/extensions/download/{report_id}/analysis - Download analysis HTML
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/extensions", tags=["Extensions"])

# Output directories
BASE_DIR = Path(__file__).resolve().parent.parent
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


# ── Request / Response models ──

class AnalyzeRequest(BaseModel):
    """Request for standalone data analysis."""
    data: str = Field(..., description="CSV or text data to analyze")


class ResearchAnalyzeRequest(BaseModel):
    """Request for combined research + analysis."""
    query: str = Field(..., description="Research query")
    provider: Optional[str] = Field(None, description="LLM provider override")
    model: Optional[str] = Field(None, description="LLM model override")


class AnalyzeResponse(BaseModel):
    """Response from analysis pipeline."""
    status: str
    extracted_data: str = ""
    data_profile: str = ""
    charts: List[str] = []
    chart_explanations: dict = {}
    outlier_analysis: Optional[str] = None
    output: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None


class ResearchAnalyzeResponse(BaseModel):
    """Response from combined research + analysis."""
    status: str
    research_report: str = ""
    analysis_report_path: str = ""
    charts: List[str] = []
    sources: List[str] = []
    sub_queries: List[str] = []
    download_urls: dict = {}
    execution_time: float = 0.0
    error: Optional[str] = None


# ── Endpoints ──

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_data(request: AnalyzeRequest):
    """Run the enforced analysis pipeline on provided data.

    Executes: extract_data → profile_data → plan charts → create charts → detect outliers
    """
    logger.info("[API] POST /api/extensions/analyze")

    try:
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.run_pipeline, request.data)

        return AnalyzeResponse(
            status=result.get("status", "error"),
            extracted_data=result.get("extracted_data", ""),
            data_profile=result.get("data_profile", ""),
            charts=result.get("charts", []),
            chart_explanations=result.get("chart_explanations", {}),
            outlier_analysis=result.get("outlier_analysis"),
            output=result.get("output", ""),
            execution_time=result.get("execution_time", 0.0),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"[API] /analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research-analyze", response_model=ResearchAnalyzeResponse)
async def research_and_analyze(request: ResearchAnalyzeRequest):
    """Run core deep research, then the analysis pipeline, then build HTML report.

    Returns both the research markdown and the analysis HTML report path.
    """
    logger.info(f"[API] POST /api/extensions/research-analyze: {request.query[:80]}...")
    report_id = str(uuid.uuid4())[:8]

    try:
        # ── Step 1: Run core deep research ──
        logger.info(f"[API] Step 1: Running deep research for report_{report_id}...")
        from langchain_core.messages import HumanMessage
        from open_deep_research.deep_researcher import deep_researcher

        config = {"configurable": {"allow_clarification": False}}
        research_result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config,
        )

        final_report = research_result.get("final_report", "")
        sources = []
        # Extract sources via regex
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', final_report)
        sources = list(set(url.rstrip(".,;:!?)") for url in urls))

        # Extract sub-queries from research plan if available
        sub_queries = []
        research_plan = research_result.get("research_plan", {})
        if isinstance(research_plan, dict):
            sub_queries = research_plan.get("subtopics", [])
        # Fallback: try notes
        if not sub_queries:
            notes = research_result.get("notes", [])
            if isinstance(notes, list) and notes:
                sub_queries = notes[:5]

        logger.info(f"[API] Step 1 complete: {len(final_report)} chars, {len(sources)} sources, {len(sub_queries)} sub-queries")

        # ── Save research markdown ──
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        research_md_path = REPORTS_DIR / f"report_{report_id}_research.md"
        research_md_path.write_text(final_report, encoding="utf-8")
        logger.info(f"[API] Research markdown saved: {research_md_path}")

        # ── Step 2: Run analysis pipeline ──
        logger.info(f"[API] Step 2: Running analysis pipeline for report_{report_id}...")
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        loop = asyncio.get_event_loop()

        # Truncate research text for analysis
        research_for_analysis = final_report
        if len(research_for_analysis) > 15000:
            research_for_analysis = research_for_analysis[:15000] + "\n\n[... truncated]"

        pipeline_result = await loop.run_in_executor(None, agent.run_pipeline, research_for_analysis)
        charts = pipeline_result.get("charts", [])
        logger.info(f"[API] Step 2 complete: {len(charts)} charts created")

        # ── Step 3: Build HTML report ──
        logger.info(f"[API] Step 3: Building HTML report for report_{report_id}...")
        from extensions.utils.report_builder import build_html_report

        analysis_report_path = build_html_report(
            display_text=final_report,
            analysis_output=pipeline_result.get("output", ""),
            figures=charts,
            chart_explanations=pipeline_result.get("chart_explanations", {}),
            sources=sources,
            query=request.query,
            sub_queries=sub_queries,
            conversation_id=report_id,
            extracted_data_summary=pipeline_result.get("extracted_data", ""),
            data_profile_summary=pipeline_result.get("data_profile", ""),
        )
        logger.info(f"[API] Step 3 complete: {analysis_report_path}")

        return ResearchAnalyzeResponse(
            status="completed",
            research_report=final_report,
            analysis_report_path=analysis_report_path,
            charts=charts,
            sources=sources,
            sub_queries=sub_queries,
            download_urls={
                "research": f"/api/extensions/download/{report_id}/research",
                "analysis": f"/api/extensions/download/{report_id}/analysis",
            },
            execution_time=pipeline_result.get("execution_time", 0.0),
        )

    except Exception as e:
        logger.error(f"[API] /research-analyze failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/{filename}")
async def serve_chart(filename: str):
    """Serve a chart HTML file from outputs/charts/."""
    chart_path = CHARTS_DIR / filename
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail=f"Chart not found: {filename}")
    return FileResponse(str(chart_path), media_type="text/html")


@router.get("/reports/{filename}")
async def serve_report(filename: str):
    """Serve an HTML report from outputs/reports/."""
    report_path = REPORTS_DIR / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")
    return FileResponse(str(report_path), media_type="text/html")


@router.get("/download/{report_id}/research")
async def download_research(report_id: str):
    """Download research markdown report."""
    md_path = REPORTS_DIR / f"report_{report_id}_research.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Research report not found: {report_id}")
    return FileResponse(
        str(md_path),
        media_type="text/markdown",
        filename=f"report_{report_id}_research.md",
    )


@router.get("/download/{report_id}/analysis")
async def download_analysis(report_id: str):
    """Download analysis HTML report."""
    html_path = REPORTS_DIR / f"report_{report_id}.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"Analysis report not found: {report_id}")
    return FileResponse(
        str(html_path),
        media_type="text/html",
        filename=f"report_{report_id}.html",
    )
