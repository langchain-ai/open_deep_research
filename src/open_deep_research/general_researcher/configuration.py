from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"

class ModelsWithWebSearch(Enum):
    ANTHROPIC_CLAUDE_3_7_SONNET_LATEST = "anthropic:claude-3-7-sonnet-latest"
    ANTHROPIC_CLAUDE_4_0_SONNET_LATEST = "anthropic:claude-sonnet-4-20250514"
    ANTHROPIC_CLAUDE_4_0_OPUS_LATEST = "anthropic:claude-opus-4-20250514"
    ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST = "anthropic:claude-3-5-haiku-latest"
    ANTHROPIC_CLAUDE_3_5_SONNET_LATEST = "anthropic:claude-3-5-sonnet-latest"
    OPENAI_GPT_4_1 = "openai:gpt-4.1"
    OPENAI_GPT_4_1_NANO = "openai:gpt-4.1-nano"

MODELS_WITH_WEB_SEARCH = {
    SearchAPI.ANTHROPIC: {
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_7_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_OPUS_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_SONNET_LATEST,
    },
    SearchAPI.OPENAI: {
        ModelsWithWebSearch.OPENAI_GPT_4_1,
        ModelsWithWebSearch.OPENAI_GPT_4_1_NANO,
    }
}

class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class WorkflowConfiguration(BaseModel):
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    clarify_with_user: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to ask the user a clarifying question before starting research"
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value}
                ]
            }
        }
    )
    search_api_config: Optional[Dict[str, Any]] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "json",
                "description": "Additional configuration for the Search API"
            }
        }
    )
    max_search_depth: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 4,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research and reflection iterations"
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="anthropic:claude-3-5-haiku-latest",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-5-haiku-latest",
                "description": "Model for summarizing research results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="anthropic:claude-sonnet-4-20250514",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-sonnet-4-20250514",
                "description": "Model for conducting research"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum tokens for research model"
            }
        }
    )
    reflection_model: str = Field(
        default="anthropic:claude-sonnet-4-20250514",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-sonnet-4-20250514",
                "description": "Model for reflecting on the current state of the research"
            }
        }
    )
    reflection_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum tokens for reflection model"
            }
        }
    )
    outliner_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for outlining the response"
            }
        }
    )
    outliner_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum tokens for outliner model"
            }
        }
    )
    final_report_model: str = Field(
        default="anthropic:claude-sonnet-4-20250514",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-sonnet-4-20250514",
                "description": "Model for final report tasks"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True