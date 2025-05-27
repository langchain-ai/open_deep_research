"""
State models for the Technical Due Diligence (TDD) Agent System.

This module defines the data models used by the TDD agents to track
their findings, evidence, and other data structures.
"""

import logging
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Enums for categorization
class RiskLevel(str, Enum):
    """Risk level for findings in due diligence."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TimelineImpact(str, Enum):
    """Impact on deal timeline."""
    NONE = "none"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    BLOCKER = "blocker"

class CostImpact(str, Enum):
    """Impact on cost."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Base models for TDD components
class Finding(BaseModel):
    """A specific finding in a domain."""
    id: str
    title: str
    description: str
    evidence: List[str]
    risk_level: RiskLevel
    timeline_impact: TimelineImpact
    cost_impact: CostImpact
    recommendations: List[str]

class Evidence(BaseModel):
    """Evidence supporting a finding."""
    id: str
    title: str
    description: str
    source: str
    content: str

class Interdependency(BaseModel):
    """Interdependency between findings."""
    id: str
    title: str
    description: str
    finding_ids: List[str]
    impact_description: str

class Gap(BaseModel):
    """Information gap requiring further investigation."""
    id: str
    title: str
    description: str
    domain: str
    questions: List[str]

class DomainReport(BaseModel):
    """Report for a specific domain."""
    id: str
    domain: str
    title: str
    summary: str
    findings: List[Finding]
    evidence: List[Evidence]
    recommendations: List[str]
    content: str
