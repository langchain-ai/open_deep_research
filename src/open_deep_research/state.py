"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic with a specialized agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
    agent_specialization: str = Field(
        description="The type of specialized agent to create for this research task. Examples: 'financial_data_analyst', 'market_researcher', 'risk_assessor', 'competitive_analyst', 'macro_economist', 'general_researcher'",
        default="general_researcher"
    )
    research_focus: str = Field(
        description="Specific focus areas for this research task. Examples: 'SEC filings and financial statements', 'industry trends and competitive landscape', 'regulatory and operational risks'",
        default="comprehensive_research"
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class QualityCheck(BaseModel):
    """Call this tool to assess the quality of research findings and provide improvement recommendations."""
    research_findings: str = Field(
        description="The research findings to assess for quality. This should include the research brief and all collected findings.",
    )

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class QualityAssessment(BaseModel):
    """Quality assessment results for equity research."""
    
    research_depth_score: int = Field(description="Score 1-5 for research depth and thoroughness")
    source_quality_score: int = Field(description="Score 1-5 for quality and credibility of sources")
    analytical_rigor_score: int = Field(description="Score 1-5 for analytical depth and reasoning")
    practical_value_score: int = Field(description="Score 1-5 for practical investment value")
    balance_objectivity_score: int = Field(description="Score 1-5 for balance and objectivity")
    writing_quality_score: int = Field(description="Score 1-5 for writing clarity and structure")
    missing_topics: str = Field(description="Specific research topics that would improve the report")
    writing_improvements: str = Field(description="Specific suggestions for improving writing style and structure")
    overall_assessment: str = Field(description="Overall assessment and summary of findings")


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    quality_assessment: str
    iteration_count: int = 0
    quality_scores: dict = {}

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    agent_specialization: str
    research_focus: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []