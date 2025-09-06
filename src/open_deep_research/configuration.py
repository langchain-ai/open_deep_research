"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class ModelPreset(Enum):
    """Enumeration of available model presets for quick configuration."""
    
    DEEPSEEK_OPENROUTER = "deepseek_openrouter"
    GPT4_OPENAI = "gpt4_openai"
    CLAUDE_ANTHROPIC = "claude_anthropic"
    GEMINI_GOOGLE = "gemini_google"
    CUSTOM = "custom"

# Model preset configurations
MODEL_PRESETS: Dict[ModelPreset, Dict[str, Any]] = {
    ModelPreset.DEEPSEEK_OPENROUTER: {
        "summarization_model": "openai:gpt-4o-mini",
        "research_model": "openai:deepseek/deepseek-chat", 
        "compression_model": "openai:deepseek/deepseek-chat",
        "final_report_model": "openai:deepseek/deepseek-chat",
        "summarization_model_max_tokens": 8192,
        "research_model_max_tokens": 10000,
        "compression_model_max_tokens": 8192,
        "final_report_model_max_tokens": 10000,
        "description": "使用 OpenRouter API 的 DeepSeek 模型，成本效益高"
    },
    ModelPreset.GPT4_OPENAI: {
        "summarization_model": "openai:gpt-4o-mini",
        "research_model": "openai:gpt-4o",
        "compression_model": "openai:gpt-4o",
        "final_report_model": "openai:gpt-4o",
        "summarization_model_max_tokens": 8192,
        "research_model_max_tokens": 8192,
        "compression_model_max_tokens": 8192,
        "final_report_model_max_tokens": 8192,
        "description": "使用 OpenAI GPT-4o 模型，性能优秀但成本较高"
    },
    ModelPreset.CLAUDE_ANTHROPIC: {
        "summarization_model": "anthropic:claude-3-5-haiku",
        "research_model": "anthropic:claude-3-5-sonnet",
        "compression_model": "anthropic:claude-3-5-sonnet",
        "final_report_model": "anthropic:claude-3-5-sonnet",
        "summarization_model_max_tokens": 8192,
        "research_model_max_tokens": 8192,
        "compression_model_max_tokens": 8192,
        "final_report_model_max_tokens": 8192,
        "description": "使用 Anthropic Claude 模型，擅长推理和分析"
    },
    ModelPreset.GEMINI_GOOGLE: {
        "summarization_model": "google:gemini-1.5-flash",
        "research_model": "google:gemini-1.5-pro",
        "compression_model": "google:gemini-1.5-pro",
        "final_report_model": "google:gemini-1.5-pro",
        "summarization_model_max_tokens": 8192,
        "research_model_max_tokens": 8192,
        "compression_model_max_tokens": 8192,
        "final_report_model_max_tokens": 8192,
        "description": "使用 Google Gemini 模型，支持长上下文"
    },
    ModelPreset.CUSTOM: {
        "description": "自定义模型配置，需要手动设置各个模型参数"
    }
}

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
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

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # Model Preset Selection
    model_preset: ModelPreset = Field(
        default=ModelPreset.DEEPSEEK_OPENROUTER,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": ModelPreset.DEEPSEEK_OPENROUTER.value,
                "description": "Choose a model preset for quick configuration. When not CUSTOM, individual model settings will be overridden.",
                "options": [
                    {"label": "DeepSeek (OpenRouter) - 成本效益", "value": ModelPreset.DEEPSEEK_OPENROUTER.value},
                    {"label": "GPT-4o (OpenAI) - 高性能", "value": ModelPreset.GPT4_OPENAI.value},
                    {"label": "Claude (Anthropic) - 善于推理", "value": ModelPreset.CLAUDE_ANTHROPIC.value},
                    {"label": "Gemini (Google) - 长上下文", "value": ModelPreset.GEMINI_GOOGLE.value},
                    {"label": "Custom - 自定义配置", "value": ModelPreset.CUSTOM.value}
                ]
            }
        }
    )
    
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
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
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
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:deepseek/deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek/deepseek-chat",
                "description": "Model for conducting research. Use 'openai:model_name' format for OpenRouter models to explicitly use OpenAI provider."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:deepseek/deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek/deepseek-chat",
                "description": "Model for compressing research findings from sub-agents. Use 'openai:model_name' format for OpenRouter models."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:deepseek/deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek/deepseek-chat",
                "description": "Model for writing the final report from all research findings. Use 'openai:model_name' format for OpenRouter models."
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
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
    apiKeys: Optional[dict[str, str]] = Field(
        default={
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
            "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY")
        },
        optional=True
    )

    @model_validator(mode='before')
    @classmethod
    def apply_model_preset(cls, data: Any) -> Any:
        """Apply model preset configuration if not using custom preset."""
        if not isinstance(data, dict):
            return data
            
        # Get the model preset from the data
        model_preset = data.get('model_preset', ModelPreset.DEEPSEEK_OPENROUTER)
        
        # If using custom preset, don't override the values
        if model_preset == ModelPreset.CUSTOM:
            return data
        
        # Ensure model_preset is a ModelPreset enum
        if isinstance(model_preset, str):
            try:
                model_preset = ModelPreset(model_preset)
            except ValueError:
                model_preset = ModelPreset.DEEPSEEK_OPENROUTER
        
        # Apply preset configuration
        preset_config = MODEL_PRESETS.get(model_preset, {})
        
        # Create a copy of data to modify
        result = data.copy()
        
        # Apply preset values for model fields
        model_fields = [
            'summarization_model', 'research_model', 'compression_model', 'final_report_model',
            'summarization_model_max_tokens', 'research_model_max_tokens', 
            'compression_model_max_tokens', 'final_report_model_max_tokens'
        ]
        
        for field_name in model_fields:
            if field_name in preset_config:
                # Only apply preset if the field is not explicitly set by user
                if field_name not in data or data[field_name] is None:
                    result[field_name] = preset_config[field_name]
                # For non-custom presets, always apply the preset (override user values)
                else:
                    result[field_name] = preset_config[field_name]
        
        return result


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
    
    def get_preset_description(self) -> str:
        """Get the description of the current model preset."""
        preset_config = MODEL_PRESETS.get(self.model_preset, {})
        return preset_config.get("description", "Unknown preset")
    
    def get_preset_info(self) -> Dict[str, Any]:
        """Get detailed information about the current model preset."""
        preset_config = MODEL_PRESETS.get(self.model_preset, {})
        return {
            "preset": self.model_preset.value,
            "description": preset_config.get("description", "Unknown preset"),
            "models": {
                "summarization": self.summarization_model,
                "research": self.research_model,
                "compression": self.compression_model,
                "final_report": self.final_report_model
            },
            "max_tokens": {
                "summarization": self.summarization_model_max_tokens,
                "research": self.research_model_max_tokens,
                "compression": self.compression_model_max_tokens,
                "final_report": self.final_report_model_max_tokens
            }
        }

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True