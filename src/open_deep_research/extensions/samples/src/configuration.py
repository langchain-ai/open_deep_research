import os
import logging
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

from enum import Enum

logger = logging.getLogger(__name__)


class SearchAPI(Enum):
    TAVILY = "tavily"


class LLMProvider(Enum):
    AZURE = "azure"
    GEMINI = "gemini"


class ActivityVerbosity(Enum):
    NONE = "none"  # No activity generation
    LOW = "low"  # Minimal activities, only for major steps
    MEDIUM = "medium"  # Moderate detail, for important transitions
    HIGH = "high"  # Detailed activities for most state changes


class Configuration:
    """The configurable fields for the research assistant."""

    def __init__(self, **kwargs):
        # Initialize from kwargs or environment variables
        self._max_web_research_loops = kwargs.get("max_web_research_loops")
        self._search_api = kwargs.get("search_api")
        self._fetch_full_page = kwargs.get("fetch_full_page")
        self._include_raw_content = kwargs.get("include_raw_content")
        self._llm_provider = kwargs.get("llm_provider")
        self._llm_model = kwargs.get("llm_model")

        # Set activity generation defaults to avoid needing .env settings
        self._enable_activity_generation = kwargs.get(
            "enable_activity_generation", True
        )
        self._activity_verbosity = kwargs.get("activity_verbosity", "medium")
        self._activity_llm_provider = kwargs.get("activity_llm_provider", "azure")
        self._activity_llm_model = kwargs.get("activity_llm_model", None)

    @property
    def max_web_research_loops(self) -> int:
        # Get the value from environment variables at runtime, not during class definition
        if self._max_web_research_loops is not None:
            return self._max_web_research_loops

        env_value = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
        print(f"Reading MAX_WEB_RESEARCH_LOOPS from environment: {env_value}")
        return int(env_value or "10")

    """
    Maximum number of web research loops to perform before finalizing.
    This helps prevent hitting the graph recursion limit (default 25).
    
    Recommended values:
    - For simple research topics: 5-8
    - For complex research topics: 8-15
    - Use values >15 with caution as you may hit recursion limits
    """

    @property
    def search_api(self):
        if self._search_api is not None:
            return self._search_api
        return SearchAPI(os.environ.get("SEARCH_API") or "tavily")

    @property
    def fetch_full_page(self) -> bool:
        if self._fetch_full_page is not None:
            return self._fetch_full_page
        return (os.environ.get("FETCH_FULL_PAGE") or "False").lower() in (
            "true",
            "1",
            "t",
        )

    @property
    def include_raw_content(self) -> bool:
        if self._include_raw_content is not None:
            return self._include_raw_content
        return (os.environ.get("INCLUDE_RAW_CONTENT") or "True").lower() in (
            "true",
            "1",
            "t",
        )

    # LLM configuration
    @property
    def llm_provider(self):
        if self._llm_provider is not None:
            return self._llm_provider
        return LLMProvider(os.environ.get("LLM_PROVIDER") or "azure")

    @property
    def llm_model(self) -> str:
        if self._llm_model is not None:
            return self._llm_model

        # Use the actual resolved provider (from frontend request or env fallback),
        # NOT the bare env LLM_PROVIDER. This prevents env LLM_PROVIDER=gemini from
        # returning "gemini-2.5-flash" when the user selected Azure from the frontend.
        resolved = self.llm_provider
        provider_str = resolved.value if isinstance(resolved, LLMProvider) else str(resolved)
        default_models = {
            "azure": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt41"),
            "gemini": "gemini-2.5-flash",
        }
        return os.environ.get("LLM_MODEL") or default_models.get(provider_str, "gpt41")

    # Activity generation configuration
    @property
    def enable_activity_generation(self) -> bool:
        """Whether to enable the generation of detailed activity descriptions."""
        if self._enable_activity_generation is not None:
            return self._enable_activity_generation
        return (os.environ.get("ENABLE_ACTIVITY_GENERATION") or "True").lower() in (
            "true",
            "1",
            "t",
        )

    @property
    def activity_verbosity(self) -> ActivityVerbosity:
        """The level of detail for generated activities."""
        if self._activity_verbosity is not None:
            return self._activity_verbosity
        verbosity_str = os.environ.get("ACTIVITY_VERBOSITY") or "medium"
        return ActivityVerbosity(verbosity_str.lower())

    @property
    def activity_llm_provider(self) -> LLMProvider:
        """The LLM provider to use for activity generation."""
        if self._activity_llm_provider is not None:
            return self._activity_llm_provider
        provider_str = os.environ.get("ACTIVITY_LLM_PROVIDER") or "azure"
        return LLMProvider(provider_str.lower())

    @property
    def activity_llm_model(self) -> str:
        """The LLM model to use for activity generation."""
        if self._activity_llm_model is not None:
            return self._activity_llm_model
        return os.environ.get("ACTIVITY_LLM_MODEL") or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt41")

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Config properties to check
        properties = [
            "max_web_research_loops",
            "search_api",
            "fetch_full_page",
            "include_raw_content",
            "llm_provider",
            "llm_model",
            "enable_activity_generation",
            "activity_verbosity",
            "activity_llm_provider",
            "activity_llm_model",
        ]

        values = {}
        for prop in properties:
            # Get from configurable or environment
            env_value = os.environ.get(prop.upper())
            config_value = configurable.get(prop)

            # Use configurable value first, then environment value
            if config_value is not None:
                values[prop] = config_value
            elif env_value is not None:
                values[prop] = env_value

        logger.info(
            f"[Configuration] from_runnable_config → "
            f"llm_provider='{values.get('llm_provider')}', "
            f"llm_model='{values.get('llm_model')}' "
            f"(configurable had: provider={configurable.get('llm_provider')}, model={configurable.get('llm_model')})"
        )

        # Create new Configuration instance with values
        return cls(**values)
