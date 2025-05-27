"""
Technology Stack tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the Tech Stack Agent to perform
technology stack assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class TechnologyStackAssessment(BaseModel):
    """Tool for assessing technology stack."""
    languages: List[str] = Field(description="Programming languages used")
    frameworks: List[str] = Field(description="Frameworks and libraries used")
    databases: List[str] = Field(description="Database technologies used")
    cloud_services: List[str] = Field(description="Cloud services used")
    development_tools: List[str] = Field(description="Development tools used")
    technical_debt_assessment: str = Field(description="Assessment of technical debt")
    scalability_assessment: str = Field(description="Assessment of scalability")
    maintainability_assessment: str = Field(description="Assessment of maintainability")
