"""
Technical Due Diligence (TDD) Agent System.

This module extends the Open Deep Research project to create a specialized
multi-agent system for technical due diligence in mergers and acquisitions.
"""

# Import agents
from open_deep_research.tdd.agents import (
    TDDAgent, PlanningAgent, TechStackAgent, ArchitectureAgent,
    SDLCAgent, InfrastructureAgent, SecurityAgent, IPAgent,
    TeamsAgent, WriterAgent
)

# Import tools
from open_deep_research.tdd.tools import (
    CreateFinding, CreateEvidence, CreateGap, CreateInterdependency,
    CreateDomainReport, TechnologyStackAssessment, ArchitectureAssessment,
    SDLCAssessment, InfrastructureAssessment, SecurityAssessment,
    IPAssessment, TeamsAssessment
)

# Import state
from open_deep_research.tdd.state.models import (
    RiskLevel, TimelineImpact, CostImpact, Finding, Evidence,
    Interdependency, Gap, DomainReport
)
from open_deep_research.tdd.state.graph_state import (
    TDDGraphState, DomainState, TDDReportState
)

# Import graph
from open_deep_research.tdd.graph import create_tdd_graph

# Import configuration
from open_deep_research.tdd.configuration import TDDConfiguration

# Import VDR
from open_deep_research.tdd.vdr import VirtualDataRoom

# Import run function
from open_deep_research.tdd.run import run_tdd

# Import integration
from open_deep_research.tdd.integration import TDDResearchAdapter

# Import pydantic for type definitions
from pydantic import BaseModel, Field

# Define input schema for LangGraph Studio
class TDDInput(BaseModel):
    query: str = Field(description="The query to run technical due diligence on")

# Define schemas for LangGraph Studio
input_schema = TDDInput
output_schema = dict

# Set up LangGraph Studio compatibility
try:
    # Define invoke function
    async def invoke(input_data):
        # Create a graph instance when needed
        graph = create_tdd_graph()
        return await graph.ainvoke({"query": input_data.query})
    
    # Flag to indicate LangGraph compatibility is available
    LANGGRAPH_COMPATIBLE = True
except ImportError:
    # LangGraph might not be available
    LANGGRAPH_COMPATIBLE = False
