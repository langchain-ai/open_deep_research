"""
Common tools for the Technical Due Diligence (TDD) Agent System.

This module defines the common tools used by all TDD agents to perform their
assessments, generate reports, and analyze findings.
"""

import logging
import uuid
from typing import List, Dict, Optional, Any, Union, Literal
from pydantic import BaseModel, Field

from langchain_core.tools import tool

from open_deep_research.tdd.state.models import (
    RiskLevel, TimelineImpact, CostImpact,
    Finding, Evidence, Interdependency, Gap, DomainReport
)

logger = logging.getLogger(__name__)

@tool
class CreateFinding(BaseModel):
    """Create a new finding in a domain."""
    title: str = Field(description="Title of the finding")
    description: str = Field(description="Detailed description of the finding")
    evidence: List[str] = Field(description="List of evidence supporting the finding")
    risk_level: RiskLevel = Field(description="Risk level of the finding")
    timeline_impact: TimelineImpact = Field(description="Impact on deal timeline")
    cost_impact: CostImpact = Field(description="Impact on cost")
    recommendations: List[str] = Field(description="Recommendations to address the finding")
    
    def get_finding(self) -> Finding:
        """Convert to a Finding object."""
        return Finding(
            id=f"finding_{uuid.uuid4().hex[:8]}",
            title=self.title,
            description=self.description,
            evidence=self.evidence,
            risk_level=self.risk_level,
            timeline_impact=self.timeline_impact,
            cost_impact=self.cost_impact,
            recommendations=self.recommendations
        )

@tool
class CreateEvidence(BaseModel):
    """Create new evidence supporting a finding."""
    title: str = Field(description="Title of the evidence")
    description: str = Field(description="Brief description of the evidence")
    source: str = Field(description="Source of the evidence (document, interview, etc.)")
    content: str = Field(description="Detailed content of the evidence")
    
    def get_evidence(self) -> Evidence:
        """Convert to an Evidence object."""
        return Evidence(
            id=f"evidence_{uuid.uuid4().hex[:8]}",
            title=self.title,
            description=self.description,
            source=self.source,
            content=self.content
        )

@tool
class CreateGap(BaseModel):
    """Create a new information gap requiring further investigation."""
    title: str = Field(description="Title of the gap")
    description: str = Field(description="Description of the information gap")
    domain: str = Field(description="Domain the gap belongs to")
    questions: List[str] = Field(description="Questions to be answered to fill the gap")
    
    def get_gap(self) -> Gap:
        """Convert to a Gap object."""
        return Gap(
            id=f"gap_{uuid.uuid4().hex[:8]}",
            title=self.title,
            description=self.description,
            domain=self.domain,
            questions=self.questions
        )

@tool
class CreateInterdependency(BaseModel):
    """Create a new interdependency between findings."""
    title: str = Field(description="Title of the interdependency")
    description: str = Field(description="Description of how the findings are related")
    finding_ids: List[str] = Field(description="IDs of the related findings")
    impact_description: str = Field(description="Description of the impact of this interdependency")
    
    def get_interdependency(self) -> Interdependency:
        """Convert to an Interdependency object."""
        return Interdependency(
            id=f"interdependency_{uuid.uuid4().hex[:8]}",
            title=self.title,
            description=self.description,
            finding_ids=self.finding_ids,
            impact_description=self.impact_description
        )

@tool
class CreateDomainReport(BaseModel):
    """Create a report for a specific domain."""
    domain: str = Field(description="Domain of the report")
    title: str = Field(description="Title of the report")
    summary: str = Field(description="Executive summary of the report")
    findings: List[Finding] = Field(description="List of findings in the domain")
    evidence: List[Evidence] = Field(description="List of evidence supporting the findings")
    recommendations: List[str] = Field(description="Overall recommendations for the domain")
    content: str = Field(description="Detailed content of the report")
    
    def get_domain_report(self) -> DomainReport:
        """Convert to a DomainReport object."""
        return DomainReport(
            id=f"report_{uuid.uuid4().hex[:8]}",
            domain=self.domain,
            title=self.title,
            summary=self.summary,
            findings=self.findings,
            evidence=self.evidence,
            recommendations=self.recommendations,
            content=self.content
        )
