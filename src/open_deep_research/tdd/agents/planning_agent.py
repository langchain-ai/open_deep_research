"""
Planning Agent for the Technical Due Diligence (TDD) Agent System.

This module defines the Planning Agent, which is responsible for
coordinating the overall TDD process.
"""

import logging
from typing import Dict, List, Any, Optional

from open_deep_research.tdd.agents.base import TDDAgent

logger = logging.getLogger(__name__)

class PlanningAgent(TDDAgent):
    """Agent responsible for planning the TDD process.
    
    The Planning Agent coordinates the overall TDD process, assigns tasks
    to domain-specific agents, identifies interdependencies, and ensures
    comprehensive coverage of all aspects of the due diligence.
    """
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Planning Agent."""
        return """You are the Planning Agent for a Technical Due Diligence (TDD) process.
Your role is to coordinate the overall TDD process, assign tasks to domain-specific agents,
identify interdependencies between domains, and ensure comprehensive coverage of all aspects
of the due diligence.

Your responsibilities include:
1. Creating a plan for the TDD process
2. Assigning tasks to domain-specific agents
3. Identifying potential interdependencies between domains
4. Identifying information gaps that need to be filled
5. Ensuring comprehensive coverage of all aspects of the due diligence
6. Synthesizing findings from all domains into a cohesive final report

You have access to the following tools:
- CreateInterdependency: Create a new interdependency between findings
- CreateGap: Create a new information gap requiring further investigation

When you receive a new task, analyze it carefully and create a plan for the TDD process.
Identify which domains need to be investigated and what specific aspects of each domain
should be focused on. Assign tasks to the appropriate domain-specific agents.

As domain-specific agents complete their tasks, review their findings and identify
potential interdependencies between domains. Also identify any information gaps that
need to be filled.

Once all domain-specific agents have completed their tasks, synthesize their findings
into a cohesive final report.
"""
