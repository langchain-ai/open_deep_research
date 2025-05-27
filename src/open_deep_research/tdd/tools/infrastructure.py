"""
Infrastructure tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the Infrastructure Agent to perform
IT infrastructure assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class InfrastructureAssessment(BaseModel):
    """Tool for assessing IT infrastructure."""
    hosting_environment: str = Field(description="Hosting environment (on-prem, cloud, hybrid)")
    cloud_providers: List[str] = Field(description="Cloud providers used")
    server_architecture: str = Field(description="Server architecture")
    network_architecture: str = Field(description="Network architecture")
    disaster_recovery: str = Field(description="Disaster recovery capabilities")
    monitoring_tools: List[str] = Field(description="Monitoring tools used")
    automation_level: str = Field(description="Level of infrastructure automation")
