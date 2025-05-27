"""
Graph state for the Technical Due Diligence (TDD) Agent System.

This module defines the state classes used by the TDD graph to track
the state of the TDD process.
"""

import logging
from typing import List, Dict, Optional, Annotated, TypedDict, Union, Any
from operator import add

from langgraph.graph import MessagesState

from open_deep_research.tdd.state.models import (
    Finding, Evidence, Interdependency, Gap, DomainReport
)

logger = logging.getLogger(__name__)

class TDDReportStateOutput(TypedDict):
    """Output state for the TDD report."""
    final_report: str

class TDDReportState(MessagesState):
    """State for the overall TDD report process."""
    domains: List[str]  # List of TDD domains to investigate
    domain_reports: Dict[str, DomainReport]  # Reports from each domain
    interdependencies: Annotated[List[Interdependency], add]  # Connections between findings
    gaps: Annotated[List[Gap], add]  # Identified information gaps
    final_report: str  # Final synthesized report

class DomainStateOutput(TypedDict):
    """Output state for a domain investigation."""
    domain_report: DomainReport

class DomainState(MessagesState):
    """State for a specific domain investigation."""
    domain: str  # Domain being investigated
    findings: Annotated[List[Finding], add]  # Specific findings in this domain
    evidence: Annotated[List[Evidence], add]  # Supporting evidence
    domain_report: Optional[DomainReport]  # Final report for this domain

class TDDGraphState(TypedDict):
    """State for the TDD graph."""
    # Query and configuration
    query: str  # The original query
    config: Dict[str, Any]  # Configuration for the TDD process
    
    # Domain tracking
    domains: List[str]  # List of domains to investigate
    current_domain: str  # Current domain being investigated
    domain_results: Dict[str, Dict[str, Any]]  # Results from each domain
    
    # Findings and evidence
    findings: List[Finding]  # All findings across domains
    evidence: List[Evidence]  # All evidence across domains
    interdependencies: List[Interdependency]  # Connections between findings
    gaps: List[Gap]  # Identified information gaps
    
    # Planning and reflection
    plan: Dict[str, Any]  # The TDD plan
    replanning_needed: bool  # Flag to indicate if replanning is needed
    replanning_consideration: bool  # Flag to consider replanning during reflection
    reflection_summary: Dict[str, Any]  # Summary of reflection
    
    # Messages
    messages: List[Any]  # Messages for the current conversation
    
    # Final report
    final_report: Optional[str]  # Final synthesized report
