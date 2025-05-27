"""
Software Development Lifecycle (SDLC) tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the SDLC Agent to perform
software development lifecycle assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class SDLCAssessment(BaseModel):
    """Tool for assessing software development lifecycle."""
    methodology: str = Field(description="Development methodology used (Agile, Waterfall, etc.)")
    version_control: str = Field(description="Version control system and practices")
    ci_cd: str = Field(description="Continuous integration and deployment practices")
    testing_practices: str = Field(description="Testing practices and coverage")
    code_review_practices: str = Field(description="Code review practices")
    release_management: str = Field(description="Release management practices")
    documentation_practices: str = Field(description="Documentation practices")
