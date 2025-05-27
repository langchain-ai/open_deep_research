"""
Architecture tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the Architecture Agent to perform
software architecture assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class ArchitectureAssessment(BaseModel):
    """Tool for assessing software architecture."""
    architecture_type: str = Field(description="Type of architecture (monolithic, microservices, etc.)")
    components: List[str] = Field(description="Main components of the architecture")
    integration_points: List[str] = Field(description="Integration points with other systems")
    data_flow: str = Field(description="Description of data flow through the system")
    scalability_design: str = Field(description="Design for scalability")
    reliability_design: str = Field(description="Design for reliability")
    security_design: str = Field(description="Design for security")
