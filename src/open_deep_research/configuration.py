"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


class VectorStoreProvider(Enum):
    """Enumeration of supported vector stores for local knowledge retrieval."""

    CHROMA = "chroma"


class EmbeddingProvider(Enum):
    """Enumeration of embedding API providers."""

    OPENAI = "openai"
    NVIDIA = "nvidia"


class MemoryWritePolicy(Enum):
    """Enumeration of write policies for long-term memory persistence."""

    EXPLICIT_CONFIRMATION = "explicit_confirmation"
    HIGH_CONFIDENCE = "high_confidence"
    ALWAYS = "always"


class EvidencePriorityStrategy(Enum):
    """Enumeration of evidence ranking strategies for multi-channel fusion."""

    LOCAL_FIRST = "local_first"
    FRESHNESS_FIRST = "freshness_first"


class MemoryMode(Enum):
    """Enumeration of memory modes exposed to API callers."""

    OFF = "off"
    SESSION_ONLY = "session_only"
    LONG_TERM_ONLY = "long_term_only"
    BOTH = "both"


class RagScope(Enum):
    """Enumeration of RAG scopes exposed to API callers."""

    DISABLED = "disabled"
    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: str | None = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: list[str] | None = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: bool | None = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
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
        default=10,#6
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=15,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 15,#10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )

    # Local RAG Configuration
    rag_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable local knowledge base retrieval (RAG)."
            }
        }
    )
    local_knowledge_base_path: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Path to local files for building/searching the knowledge base (md/txt/pdf/docx)."
            }
        }
    )
    vector_store_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.CHROMA,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": VectorStoreProvider.CHROMA.value,
                "description": "Vector store used for local knowledge retrieval.",
                "options": [
                    {"label": "Chroma", "value": VectorStoreProvider.CHROMA.value}
                ]
            }
        }
    )
    chroma_persist_directory: str = Field(
        default=".chroma",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".chroma",
                "description": "Persistent storage directory for the Chroma vector store."
            }
        }
    )
    embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:text-embedding-3-small",
                "description": "Embedding model used to index and retrieve local knowledge."
            }
        }
    )
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": EmbeddingProvider.OPENAI.value,
                "description": "Embedding API provider used for local RAG.",
                "options": [
                    {"label": "OpenAI Compatible", "value": EmbeddingProvider.OPENAI.value},
                    {"label": "NVIDIA NIM", "value": EmbeddingProvider.NVIDIA.value}
                ]
            }
        }
    )
    embedding_encoding_format: str = Field(
        default="float",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "float",
                "description": "Optional embedding encoding format. For NVIDIA, use float."
            }
        }
    )
    embedding_input_type_query: str = Field(
        default="query",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "query",
                "description": "Input type used when embedding retrieval queries (NVIDIA)."
            }
        }
    )
    embedding_input_type_document: str = Field(
        default="passage",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "passage",
                "description": "Input type used when embedding indexed documents (NVIDIA)."
            }
        }
    )
    embedding_truncate: str = Field(
        default="NONE",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "NONE",
                "description": "Embedding truncation mode for providers that require explicit truncation behavior."
            }
        }
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Number of retrieved chunks to return from local knowledge search."
            }
        }
    )

    # Personal Memory Configuration
    memory_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable short-term and long-term user memory in research workflows."
            }
        }
    )
    memory_write_policy: MemoryWritePolicy = Field(
        default=MemoryWritePolicy.EXPLICIT_CONFIRMATION,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": MemoryWritePolicy.EXPLICIT_CONFIRMATION.value,
                "description": "When long-term memory can be persisted for a user.",
                "options": [
                    {
                        "label": "Explicit confirmation",
                        "value": MemoryWritePolicy.EXPLICIT_CONFIRMATION.value
                    },
                    {
                        "label": "High confidence only",
                        "value": MemoryWritePolicy.HIGH_CONFIDENCE.value
                    },
                    {
                        "label": "Always",
                        "value": MemoryWritePolicy.ALWAYS.value
                    }
                ]
            }
        }
    )
    memory_max_candidates_per_turn: int = Field(
        default=3,
        ge=1,
        le=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum memory candidates proposed per turn for long-term persistence."
            }
        }
    )
    memory_namespace_prefix: str = Field(
        default="memory",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "memory",
                "description": "Namespace prefix used when storing user memory in persistent store."
            }
        }
    )
    user_id: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Optional user identifier for local testing when metadata.owner is not provided."
            }
        }
    )
    evidence_priority_strategy: EvidencePriorityStrategy = Field(
        default=EvidencePriorityStrategy.LOCAL_FIRST,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": EvidencePriorityStrategy.LOCAL_FIRST.value,
                "description": "Evidence prioritization strategy when fusing web, local RAG, and memory channels.",
                "options": [
                    {"label": "Local first", "value": EvidencePriorityStrategy.LOCAL_FIRST.value},
                    {"label": "Freshness first", "value": EvidencePriorityStrategy.FRESHNESS_FIRST.value}
                ]
            }
        }
    )
    memory_mode: MemoryMode = Field(
        default=MemoryMode.BOTH,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": MemoryMode.BOTH.value,
                "description": "Runtime memory mode for API calls.",
                "options": [
                    {"label": "Off", "value": MemoryMode.OFF.value},
                    {"label": "Session only", "value": MemoryMode.SESSION_ONLY.value},
                    {"label": "Long-term only", "value": MemoryMode.LONG_TERM_ONLY.value},
                    {"label": "Both", "value": MemoryMode.BOTH.value}
                ]
            }
        }
    )
    rag_scope: RagScope = Field(
        default=RagScope.HYBRID,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": RagScope.HYBRID.value,
                "description": "Runtime RAG scope for API calls.",
                "options": [
                    {"label": "Disabled", "value": RagScope.DISABLED.value},
                    {"label": "Local only", "value": RagScope.LOCAL_ONLY.value},
                    {"label": "Hybrid", "value": RagScope.HYBRID.value}
                ]
            }
        }
    )
    api_include_progress: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Include progress section in API response payload."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:meta/llama-3.3-70b-instruct",  # <-- 改这里
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:meta/llama-3.3-70b-instruct", # <-- 改这里 (UI默认值也顺手改了)
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=16384, #8192
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 16384,#8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=80000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 80000 , # 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:meta/llama-3.3-70b-instruct", # <-- 改这里
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:meta/llama-3.3-70b-instruct", # <-- 改这里
                "description": "Model for conducting research..."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=16000, #10000
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 16000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:meta/llama-3.3-70b-instruct", # <-- 改这里
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:meta/llama-3.3-70b-instruct", # <-- 改这里
                "description": "Model for compressing research findings..."
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
        default="openai:meta/llama-3.3-70b-instruct", # <-- 改这里
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:meta/llama-3.3-70b-instruct", # <-- 改这里
                "description": "Model for writing the final report..."
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=20000, #10000
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 20000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: MCPConfig | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: str | None = Field(
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
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    @model_validator(mode="after")
    def validate_personalization_settings(self) -> "Configuration":
        """Validate the minimum constraints for RAG and memory settings."""
        if self.rag_enabled and not self.chroma_persist_directory.strip():
            raise ValueError(
                "chroma_persist_directory must be provided when rag_enabled is true"
            )

        if self.rag_enabled:
            try:
                persist_dir = Path(self.chroma_persist_directory).expanduser()
            except Exception as exc:
                raise ValueError(
                    "chroma_persist_directory is not a valid filesystem path"
                ) from exc

            parent_dir = persist_dir.parent if persist_dir.parent != Path("") else Path(".")
            if parent_dir.exists() and not parent_dir.is_dir():
                raise ValueError(
                    "chroma_persist_directory parent path exists but is not a directory"
                )

        if self.memory_enabled and not self.memory_namespace_prefix.strip():
            raise ValueError(
                "memory_namespace_prefix must be provided when memory_enabled is true"
            )

        if self.embedding_provider == EmbeddingProvider.NVIDIA:
            if not self.embedding_input_type_query.strip():
                raise ValueError(
                    "embedding_input_type_query must be provided when embedding_provider is nvidia"
                )
            if not self.embedding_input_type_document.strip():
                raise ValueError(
                    "embedding_input_type_document must be provided when embedding_provider is nvidia"
                )

        return self

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True