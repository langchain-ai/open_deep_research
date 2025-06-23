from typing import Annotated, List, Literal, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation

# Structured output
class ClarifyWithUser(BaseModel):
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    content: str = Field(
        description="The content of the section."
    )

class Outline(BaseModel):
    outline: List[Section] = Field(
        description="List of sections for the report.",
    )

class SearchSource(BaseModel):
    """Represents a source found during web search."""
    url: str
    title: str
    content: str
    search_query: str
    iteration: int

class ReflectionResult(BaseModel):
    """Result of reflection phase analyzing research completeness."""
    is_satisfied: bool = Field(
        description="Whether the research is complete enough to move on to the next phase.",
    )
    knowledge_gaps: List[str] = Field(
        description="List of knowledge gaps that need to be filled in with further research. This can be empty if is_satisfied is true.",
    )
    suggested_queries: List[str] = Field(
        description="List of suggested queries to search for to fill the knowledge gaps. This can be empty if is_satisfied is true.s",
    )
    reasoning: str = Field(
        description="Reasoning for the reflection result.",
    )

# State Definitions
class ResearchUnitState(MessagesState):
    notes: Annotated[Optional[list[str]], operator.add]
    research_iterations: Optional[int] = 0
    research_messages: Optional[list[MessageLikeRepresentation]]
    collected_sources: Annotated[Optional[list[SearchSource]], operator.add]

class GeneralResearcherStateInput(MessagesState):
    """InputState is only 'messages'"""

class GeneralResearcherStateOutput(MessagesState):
    final_report: str

class GeneralResearcherState(MessagesState):
    final_report: str
    notes: Annotated[list[str], operator.add]
    collected_sources: Optional[list[SearchSource]]
    search_attempts: int = 0
    outline: list[Section]
    completed_sections: Annotated[list, operator.add]