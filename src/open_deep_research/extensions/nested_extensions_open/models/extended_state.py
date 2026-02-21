"""Extended state for master agent - does NOT modify open_deep_research state."""
from typing import TypedDict, Optional, List, Dict
from typing_extensions import NotRequired


class MasterAgentState(TypedDict, total=False):
    """Master agent state for orchestrating multiple agents.
    
    This is SEPARATE from open_deep_research's AgentState.
    Used to track multi-agent orchestration results.
    """
    
    # =====================================
    # CORE METADATA
    # =====================================
    conversation_id: str                  # Unique ID (UUID)
    session_id: NotRequired[str]          # Session ID for grouping conversations
    query: str                            # Original user query
    status: str                           # "pending" | "running" | "completed" | "error"
    timestamp: str                        # ISO timestamp
    
    # =====================================
    # RESEARCH RESULTS (from open_deep_research)
    # =====================================
    final_report: NotRequired[str]        # Research report from deep_researcher
    sources: NotRequired[List[str]]       # URLs/sources used
    sources_text: NotRequired[List[str]]  # Formatted sources for display (markdown links)
    sub_queries: NotRequired[List[str]]   # Sub-queries used in enhanced research
    
    # =====================================
    # ANALYSIS RESULTS
    # =====================================
    analysis_output: NotRequired[str]     # Data analysis summary
    charts: NotRequired[List[str]]        # Chart file paths
    chart_explanations: NotRequired[Dict[str, Dict[str, str]]]  # {path: {title, explanation}}
    extracted_data: NotRequired[str]      # Extracted/structured data (CSV tables from pipeline)
    data_profile: NotRequired[str]        # Data profile output from pipeline

    # =====================================
    # GENERATED REPORTS
    # =====================================
    report_html: NotRequired[str]         # HTML report with embedded plots
    
    # =====================================
    # ORCHESTRATION
    # =====================================
    agents_used: NotRequired[List[str]]   # List of agents invoked
    execution_time: NotRequired[float]    # Total execution time (seconds)
    
    # =====================================
    # ERROR
    # =====================================
    error: NotRequired[str]               # Error message if any


__all__ = ['MasterAgentState']
