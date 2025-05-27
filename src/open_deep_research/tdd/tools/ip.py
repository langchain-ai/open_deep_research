"""
Intellectual Property (IP) tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the IP Agent to perform
technical intellectual property assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class IPAssessment(BaseModel):
    """Tool for assessing technical intellectual property."""
    proprietary_technologies: List[str] = Field(description="Proprietary technologies")
    patents: List[str] = Field(description="Patents owned or pending")
    open_source_usage: str = Field(description="Open source usage and compliance")
    licensing_issues: List[str] = Field(description="Licensing issues identified")
    ip_protection_measures: str = Field(description="IP protection measures")
    third_party_dependencies: List[str] = Field(description="Third-party dependencies")
    ip_risks: List[str] = Field(description="IP-related risks")
