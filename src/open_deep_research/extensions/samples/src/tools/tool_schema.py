"""
Tool schemas for the research agent's search tools.

This module provides structured tool schemas for use with LLM function calling APIs.
These schemas define the search tools available to the research agent in a format
that can be directly used with function calling in modern LLMs.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Literal, Union, Callable
from pydantic import BaseModel, Field


# Tool schema for generic tools (including MCP tools)
class ToolParameterType(str, Enum):
    """Enum for tool parameter types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class ToolParameter:
    """Parameter definition for a tool."""

    def __init__(
        self,
        name: str,
        type: ToolParameterType = ToolParameterType.STRING,
        required: bool = True,
        description: str = "",
    ):
        """Initialize a tool parameter.

        Args:
            name: Name of the parameter
            type: Type of the parameter
            required: Whether the parameter is required
            description: Description of the parameter
        """
        self.name = name
        self.type = type
        self.required = required
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter to a dictionary.

        Returns:
            Dictionary representation of the parameter
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "required": self.required,
            "description": self.description,
        }


class Tool:
    """Generic tool that can be executed by the executor."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Optional[Callable[..., Any]] = None,
        lc_tool: Optional[Any] = None,
    ):
        """Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of the tool
            parameters: List of parameters for the tool
            function: Function to execute when the tool is called (for non-LC tools)
            lc_tool: LangChain tool instance (for LC/MCP tools)
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.lc_tool = lc_tool
        self.config = None  # Will be set by registry if needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary.

        Returns:
            Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.to_dict() for param in self.parameters],
        }


class GeneralSearchToolSchema(BaseModel):
    """Schema for the general search tool."""

    query: str = Field(
        ..., description="The search query string (5-10 keywords) to execute"
    )

    class Config:
        schema_extra = {
            "name": "general_search",
            "description": "For broad topics that don't fit specialized categories. Best for general information, news, consumer topics, and multifaceted questions. Default tool when no specialized tool is clearly better.",
        }


class AcademicSearchToolSchema(BaseModel):
    """Schema for the academic search tool."""

    query: str = Field(
        ..., description="The search query string (5-10 keywords) to execute"
    )

    class Config:
        schema_extra = {
            "name": "academic_search",
            "description": "For scholarly/academic/scientific topics. Best for research papers, scientific studies, academic publications. Useful for technical specifications, methodologies, and peer-reviewed content. Suggested when topics mention papers, research, studies, science, or scholarly content.",
        }


class GithubSearchToolSchema(BaseModel):
    """Schema for the Github search tool."""

    query: str = Field(
        ..., description="The search query string (5-10 keywords) to execute"
    )

    class Config:
        schema_extra = {
            "name": "github_search",
            "description": "For code, programming, and software development topics. Best for finding repositories, code examples, software implementations. Useful for technical documentation and open-source projects. Suggested when topics mention code, programming, development, repositories, or software.",
        }


class LinkedinSearchToolSchema(BaseModel):
    """Schema for the LinkedIn search tool."""

    query: str = Field(
        ..., description="The search query string (5-10 keywords) to execute"
    )

    class Config:
        schema_extra = {
            "name": "linkedin_search",
            "description": "For professional and people-related topics. Best for finding information about individuals, companies, and professional backgrounds. Useful for industry experts, leadership information, and professional profiles. Suggested when topics mention people, professionals, executives, or company leadership.",
        }


# Function specifications that can be used directly with OpenAI/LLM function calling
SEARCH_TOOL_FUNCTIONS = [
    {
        "name": "general_search",
        "description": "For broad topics that don't fit specialized categories. Best for general information, news, consumer topics, and multifaceted questions. Default tool when no specialized tool is clearly better.",
        "parameters": GeneralSearchToolSchema.schema(),
    },
    {
        "name": "academic_search",
        "description": "For scholarly/academic/scientific topics. Best for research papers, scientific studies, academic publications. Useful for technical specifications, methodologies, and peer-reviewed content. Suggested when topics mention papers, research, studies, science, or scholarly content.",
        "parameters": AcademicSearchToolSchema.schema(),
    },
    {
        "name": "github_search",
        "description": "For code, programming, and software development topics. Best for finding repositories, code examples, software implementations. Useful for technical documentation and open-source projects. Suggested when topics mention code, programming, development, repositories, or software.",
        "parameters": GithubSearchToolSchema.schema(),
    },
    {
        "name": "linkedin_search",
        "description": "For professional and people-related topics. Best for finding information about individuals, companies, and professional backgrounds. Useful for industry experts, leadership information, and professional profiles. Suggested when topics mention people, professionals, executives, or company leadership.",
        "parameters": LinkedinSearchToolSchema.schema(),
    },
    {
        "name": "text2sql",
        "description": "For querying uploaded databases with natural language. Best for data analysis, business intelligence, and extracting insights from structured data. Useful when users want to analyze data, generate reports, or answer questions about their uploaded database files. Suggested when topics mention data analysis, database queries, business metrics, or when users have uploaded database files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to convert to SQL and execute against uploaded databases",
                },
                "db_id": {
                    "type": "string",
                    "description": "Optional database ID to query. If not provided, uses the first available database.",
                },
            },
            "required": ["query"],
        },
    },
]


# Simple topic response schema
class SimpleTopicResponse(BaseModel):
    """Response schema for simple topics."""

    query: str = Field(
        ..., description="The actual search query string (under 400 characters)"
    )
    aspect: str = Field(..., description="The specific aspect or angle of the topic")
    rationale: str = Field(
        ..., description="Brief explanation of why this query is relevant"
    )
    suggested_tool: Literal[
        "general_search",
        "academic_search",
        "github_search",
        "linkedin_search",
        "text2sql",
    ] = Field(..., description="Most appropriate search tool for this query")


# Subtopic schema for complex topics
class Subtopic(BaseModel):
    """Schema for a subtopic in a complex topic."""

    name: str = Field(..., description="Brief name of the subtopic")
    query: str = Field(
        ...,
        description="Specific search query for this subtopic (under 400 characters)",
    )
    aspect: str = Field(..., description="The specific aspect this subtopic covers")
    suggested_tool: Literal[
        "general_search",
        "academic_search",
        "github_search",
        "linkedin_search",
        "text2sql",
    ] = Field(..., description="Most appropriate search tool for this subtopic")


# Complex topic response schema
class ComplexTopicResponse(BaseModel):
    """Response schema for complex topics."""

    topic_complexity: Literal["complex"] = Field(
        "complex",
        description="Indicates this is a complex topic requiring multiple subtopics",
    )
    main_query: str = Field(
        ..., description="Single search query to use if a backup is needed"
    )
    main_tool: Literal[
        "general_search",
        "academic_search",
        "github_search",
        "linkedin_search",
        "text2sql",
    ] = Field(..., description="Suggested search tool for the main query")
    subtopics: List[Subtopic] = Field(
        ...,
        description="List of 3-5 subtopics that together provide comprehensive coverage",
    )


# Combined response schema for topic decomposition
class TopicDecompositionResponse(BaseModel):
    """Schema for the topic decomposition response."""

    response: Union[SimpleTopicResponse, ComplexTopicResponse] = Field(
        ..., description="Either a simple topic response or a complex topic response"
    )


# Define subtopic schema structure explicitly for JSON schema compliance
subtopic_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Brief name of the subtopic"},
        "query": {
            "type": "string",
            "description": "Specific search query for this subtopic (under 400 characters)",
        },
        "aspect": {
            "type": "string",
            "description": "The specific aspect this subtopic covers",
        },
        "suggested_tool": {
            "type": "string",
            "enum": [
                "general_search",
                "academic_search",
                "github_search",
                "linkedin_search",
                "text2sql",
            ],
            "description": "Most appropriate search tool for this subtopic",
        },
    },
    "required": ["name", "query", "aspect", "suggested_tool"],
}

# Function specification for topic decomposition (Using 'parameters' key for Langchain compatibility)
TOPIC_DECOMPOSITION_FUNCTION = {
    "name": "decompose_research_topic",
    "description": "Analyze a research topic and either decompose it into subtopics if complex, or create a targeted search query if simple.",
    "parameters": {  # Use 'parameters' for Langchain's bind_tools
        "type": "object",
        "properties": {
            "topic_complexity": {
                "type": "string",
                "enum": ["simple", "complex"],
                "description": "Whether this is a simple topic (single query) or complex topic (multiple subtopics)",
            },
            "simple_topic": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The actual search query string (under 400 characters)",
                    },
                    "aspect": {
                        "type": "string",
                        "description": "The specific aspect or angle of the topic",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief explanation of why this query is relevant",
                    },
                    "suggested_tool": {
                        "type": "string",
                        "enum": [
                            "general_search",
                            "academic_search",
                            "github_search",
                            "linkedin_search",
                            "text2sql",
                        ],
                        "description": "Most appropriate search tool for this query",
                    },
                },
                "required": ["query", "aspect", "rationale", "suggested_tool"],
                "description": "Response details for a simple topic.",
            },
            "complex_topic": {
                "type": "object",
                "properties": {
                    "main_query": {
                        "type": "string",
                        "description": "Single search query to use if a backup is needed",
                    },
                    "main_tool": {
                        "type": "string",
                        "enum": [
                            "general_search",
                            "academic_search",
                            "github_search",
                            "linkedin_search",
                            "text2sql",
                        ],
                        "description": "Suggested search tool for the main query",
                    },
                    "subtopics": {
                        "type": "array",
                        "items": subtopic_schema,  # Use the explicitly defined subtopic schema
                        "description": "List of 3-5 subtopics that together provide comprehensive coverage",
                    },
                },
                "required": ["main_query", "main_tool", "subtopics"],
                "description": "Response details for a complex topic.",
            },
        },
        "required": ["topic_complexity"],
    },
}
