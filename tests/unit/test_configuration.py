"""Unit tests for open_deep_research/configuration.py.

Tests cover Configuration model defaults, from_runnable_config loading,
SearchAPI enum values, and MCPConfig defaults.
"""

import os
from unittest.mock import patch

import pytest

from open_deep_research.configuration import Configuration, MCPConfig, SearchAPI


class TestSearchAPI:
    """Tests for the SearchAPI enum."""

    def test_all_expected_values_exist(self):
        assert SearchAPI.TAVILY.value == "tavily"
        assert SearchAPI.OPENAI.value == "openai"
        assert SearchAPI.ANTHROPIC.value == "anthropic"
        assert SearchAPI.NONE.value == "none"

    def test_enum_member_count(self):
        """Ensure no enum members were accidentally removed."""
        assert len(SearchAPI) == 4


class TestMCPConfig:
    """Tests for MCPConfig model defaults."""

    def test_defaults(self):
        config = MCPConfig()
        assert config.url is None
        assert config.tools is None
        assert config.auth_required is False

    def test_with_values(self):
        config = MCPConfig(
            url="https://mcp.example.com",
            tools=["search", "browse"],
            auth_required=True,
        )
        assert config.url == "https://mcp.example.com"
        assert config.tools == ["search", "browse"]
        assert config.auth_required is True


class TestConfiguration:
    """Tests for the main Configuration model."""

    def test_defaults(self):
        """All defaults should match the documented values."""
        config = Configuration()

        assert config.max_structured_output_retries == 3
        assert config.allow_clarification is True
        assert config.max_concurrent_research_units == 5
        assert config.search_api == SearchAPI.TAVILY
        assert config.max_researcher_iterations == 6
        assert config.max_react_tool_calls == 10
        assert config.summarization_model == "openai:gpt-4.1-mini"
        assert config.summarization_model_max_tokens == 8192
        assert config.max_content_length == 50000
        assert config.research_model == "openai:gpt-4.1"
        assert config.research_model_max_tokens == 10000
        assert config.compression_model == "openai:gpt-4.1"
        assert config.compression_model_max_tokens == 8192
        assert config.final_report_model == "openai:gpt-4.1"
        assert config.final_report_model_max_tokens == 10000
        assert config.mcp_config is None
        assert config.mcp_prompt is None

    def test_from_runnable_config_with_overrides(self):
        """Configuration should be loadable from a RunnableConfig dict."""
        runnable_config = {
            "configurable": {
                "research_model": "anthropic:claude-sonnet-4",
                "max_researcher_iterations": 10,
                "search_api": "openai",
            }
        }
        config = Configuration.from_runnable_config(runnable_config)

        assert config.research_model == "anthropic:claude-sonnet-4"
        assert config.max_researcher_iterations == 10
        # Default values should still be present for non-overridden fields
        assert config.summarization_model == "openai:gpt-4.1-mini"

    def test_from_runnable_config_empty(self):
        """Empty configurable should produce all defaults."""
        config = Configuration.from_runnable_config({"configurable": {}})
        assert config.research_model == "openai:gpt-4.1"
        assert config.search_api == SearchAPI.TAVILY

    def test_from_runnable_config_none(self):
        """None config should produce all defaults."""
        config = Configuration.from_runnable_config(None)
        assert config.research_model == "openai:gpt-4.1"

    @patch.dict(os.environ, {"RESEARCH_MODEL": "openai:o3-mini"})
    def test_from_runnable_config_env_override(self):
        """Environment variables should take precedence."""
        config = Configuration.from_runnable_config({"configurable": {}})
        assert config.research_model == "openai:o3-mini"

    @patch.dict(os.environ, {"SEARCH_API": "anthropic"})
    def test_from_runnable_config_env_search_api(self):
        """Search API should be loadable from environment."""
        config = Configuration.from_runnable_config({"configurable": {}})
        assert config.search_api == SearchAPI.ANTHROPIC

    def test_custom_model_configuration(self):
        """Configuration should accept any model string."""
        config = Configuration(
            research_model="groq:llama-3.3-70b",
            compression_model="deepseek:deepseek-chat",
            final_report_model="mistral:mistral-large",
        )
        assert config.research_model == "groq:llama-3.3-70b"
        assert config.compression_model == "deepseek:deepseek-chat"
        assert config.final_report_model == "mistral:mistral-large"
