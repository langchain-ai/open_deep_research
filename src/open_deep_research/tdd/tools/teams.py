"""
Teams tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the Teams Agent to perform
technical teams and processes assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class TeamsAssessment(BaseModel):
    """Tool for assessing technical teams and processes."""
    team_structure: str = Field(description="Team structure and organization")
    key_personnel: List[str] = Field(description="Key technical personnel")
    skills_assessment: str = Field(description="Assessment of team skills")
    knowledge_sharing: str = Field(description="Knowledge sharing practices")
    onboarding_processes: str = Field(description="Onboarding processes")
    retention_risks: List[str] = Field(description="Retention risks identified")
    culture_assessment: str = Field(description="Assessment of technical culture")
