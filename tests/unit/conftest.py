"""Shared test fixtures for unit tests."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def simple_messages():
    """A simple message history: Human → AI → Tool → AI → Tool → Human."""
    return [
        HumanMessage(content="Research quantum computing"),
        AIMessage(
            content="I'll search for quantum computing info",
            tool_calls=[{"name": "tavily_search", "args": {"queries": ["quantum computing"]}, "id": "tc1"}],
        ),
        ToolMessage(content="Quantum computing uses qubits...", name="tavily_search", tool_call_id="tc1"),
        AIMessage(
            content="Let me search for more details",
            tool_calls=[{"name": "tavily_search", "args": {"queries": ["quantum supremacy"]}, "id": "tc2"}],
        ),
        ToolMessage(content="Google achieved quantum supremacy in 2019...", name="tavily_search", tool_call_id="tc2"),
        HumanMessage(content="Please compress your findings"),
    ]


@pytest.fixture
def no_ai_messages():
    """A message list with no AI messages."""
    return [
        HumanMessage(content="Hello"),
        HumanMessage(content="World"),
    ]


@pytest.fixture
def single_ai_exchange():
    """A message list with a single AI→Tool exchange."""
    return [
        HumanMessage(content="Search for X"),
        AIMessage(
            content="Searching...",
            tool_calls=[{"name": "tavily_search", "args": {"queries": ["X"]}, "id": "tc1"}],
        ),
        ToolMessage(content="Results for X", name="tavily_search", tool_call_id="tc1"),
    ]


@pytest.fixture
def multi_tool_ai_exchange():
    """An AI message followed by multiple tool messages."""
    return [
        HumanMessage(content="Research topic"),
        AIMessage(
            content="I'll run multiple searches",
            tool_calls=[
                {"name": "tavily_search", "args": {"queries": ["topic A"]}, "id": "tc1"},
                {"name": "tavily_search", "args": {"queries": ["topic B"]}, "id": "tc2"},
            ],
        ),
        ToolMessage(content="Results for A", name="tavily_search", tool_call_id="tc1"),
        ToolMessage(content="Results for B", name="tavily_search", tool_call_id="tc2"),
        AIMessage(content="Here are the final findings"),
    ]


@pytest.fixture
def mock_runnable_config():
    """A mock RunnableConfig with configurable values."""
    return {
        "configurable": {
            "search_api": "tavily",
            "research_model": "openai:gpt-4.1",
            "summarization_model": "openai:gpt-4.1-mini",
            "compression_model": "anthropic:claude-sonnet-4",
            "final_report_model": "openai:gpt-4.1",
        }
    }


@pytest.fixture
def mock_runnable_config_with_api_keys():
    """A mock RunnableConfig with API keys in configurable."""
    return {
        "configurable": {
            "apiKeys": {
                "OPENAI_API_KEY": "sk-test-openai",
                "ANTHROPIC_API_KEY": "sk-test-anthropic",
                "GOOGLE_API_KEY": "test-google-key",
                "GROQ_API_KEY": "test-groq-key",
                "DEEPSEEK_API_KEY": "test-deepseek-key",
                "MISTRAL_API_KEY": "test-mistral-key",
                "COHERE_API_KEY": "test-cohere-key",
                "FIREWORKS_API_KEY": "test-fireworks-key",
                "AWS_ACCESS_KEY_ID": "test-aws-key",
            }
        }
    }
