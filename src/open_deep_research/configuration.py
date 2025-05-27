import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, ClassVar, Type, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    """Enumeration of supported search APIs.
    
    This enum defines all the search APIs that can be used for research.
    The value of each enum member is the string identifier used in configuration.
    """
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """Configuration class for the research system.
    
    This class defines all configurable parameters for both the graph-based and
    multi-agent implementations of the research system. It provides default values
    for all parameters and methods to create a Configuration instance from a
    RunnableConfig.
    """
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE  # Structure template for the report
    search_api: SearchAPI = SearchAPI.TAVILY  # Search API to use for research
    search_api_config: Optional[Dict[str, Any]] = None  # Additional configuration for search API
    
    # Graph-specific configuration
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    planner_provider: str = "groq"  # Provider for the planner model
    planner_model: str = "groq:llama-3.3-70b-versatile"  # Model for planning report sections
    planner_model_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for planner model
    writer_provider: str = "groq"  # Provider for the writer model
    writer_model: str = "groq:llama-3.3-70b-versatile"  # Model for writing report sections
    writer_model_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for writer model
    
    # Multi-agent specific configuration
    supervisor_model: str = "groq:llama-3.3-70b-versatile"  # Model for supervisor agent
    researcher_model: str = "groq:llama-3.3-70b-versatile"  # Model for research agents

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.
        
        This method extracts configuration values from either environment variables
        or the configurable section of a RunnableConfig. Environment variables take
        precedence over values in the RunnableConfig.
        
        Args:
            config: Optional RunnableConfig containing configuration values
            
        Returns:
            A new Configuration instance with values from the RunnableConfig and/or
            environment variables
        """
        # Extract configurable dict from RunnableConfig if it exists
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # For each field in the Configuration class, try to get its value from
        # environment variables first, then from the configurable dict
        values: Dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        
        # Create a new Configuration instance with non-None values
        return cls(**{k: v for k, v in values.items() if v is not None})
