import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Literal

from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list or table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"

@dataclass(kw_only=True)
class WorkflowConfiguration:
    """Configuration for the workflow/graph-based implementation (graph.py)."""
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Dict[str, Any] | None = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "google_genai"  # Options: "anthropic", "openai", "google_genai", "groq", "deepseek", etc.
    summarization_model: str = "gemini-2.5-flash"  # For Google: "gemini-1.5-flash", "gemini-1.5-pro", etc.
    max_structured_output_retries: int = 3
    include_source_str: bool = False
    
    # Workflow-specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "google_genai"  # Options: "anthropic", "openai", "google_genai", "groq", "deepseek", etc.
    planner_model: str = "gemini-2.5-flash"  # For Google: "gemini-1.5-pro", "gemini-1.5-flash", etc.
    planner_model_kwargs: Dict[str, Any] | None = None
    writer_provider: str = "google_genai"  # Options: "anthropic", "openai", "google_genai", "groq", "deepseek", etc.
    writer_model: str = "gemini-2.5-flash"  # For Google: "gemini-1.5-pro", "gemini-1.5-flash", etc.
    writer_model_kwargs: Dict[str, Any] | None = None

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

@dataclass(kw_only=True)
class MultiAgentConfiguration:
    """Configuration for the multi-agent implementation (multi_agent.py)."""
    # Common configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Dict[str, Any] | None = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "google_genai"  # Options: "anthropic", "openai", "google_genai", "groq", "deepseek", etc.
    summarization_model: str = "gemini-2.5-flash"  # For Google: "gemini-1.5-flash", "gemini-1.5-pro", etc.
    include_source_str: bool = False
    
    # Multi-agent specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per section
    supervisor_model: str = "google_genai:gemini-2.5-flash"  # Options: "google_genai:gemini-1.5-pro", "openai:gpt-4o", etc.
    researcher_model: str = "google_genai:gemini-2.5-flash"  # Options: "google_genai:gemini-1.5-flash", "openai:gpt-4o", etc.
    ask_for_clarification: bool = False # Whether to ask for clarification from the user
    # MCP server configuration
    mcp_server_config: Dict[str, Any] | None = None
    mcp_prompt: str | None = None
    mcp_tools_to_include: list[str] | None = None

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# Keep the old Configuration class for backward compatibility
Configuration = WorkflowConfiguration
