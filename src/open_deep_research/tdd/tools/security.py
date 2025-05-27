"""
Security tools for the Technical Due Diligence (TDD) Agent System.

This module defines the tools used by the Security Agent to perform
cybersecurity assessments.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
class SecurityAssessment(BaseModel):
    """Tool for assessing cybersecurity."""
    vulnerabilities: List[str] = Field(description="Identified vulnerabilities")
    security_controls: List[str] = Field(description="Security controls in place")
    compliance_status: str = Field(description="Compliance status with relevant standards")
    authentication_mechanisms: str = Field(description="Authentication mechanisms")
    authorization_mechanisms: str = Field(description="Authorization mechanisms")
    data_protection: str = Field(description="Data protection measures")
    incident_response: str = Field(description="Incident response capabilities")
