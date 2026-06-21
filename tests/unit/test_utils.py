"""Comprehensive unit tests for open_deep_research/utils.py.

Tests cover all pure-logic utility functions that don't require API calls:
- remove_up_to_last_ai_message (token recovery)
- is_token_limit_exceeded (error detection across providers)
- get_model_token_limit (model lookup)
- get_config_value (enum/string/dict extraction)
- get_today_str (date formatting)
- anthropic_websearch_called / openai_websearch_called (response parsing)
- get_api_key_for_model (provider API key resolution)
- get_notes_from_tool_calls (tool message extraction)
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from open_deep_research.configuration import SearchAPI
from open_deep_research.utils import (
    _check_anthropic_token_limit,
    _check_gemini_token_limit,
    _check_openai_token_limit,
    anthropic_websearch_called,
    get_api_key_for_model,
    get_config_value,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
)


# ===================================================================
# remove_up_to_last_ai_message — Bug #252 fix tests
# ===================================================================

class TestRemoveUpToLastAIMessage:
    """Tests for the token recovery function that removes oldest AI exchanges."""

    def test_removes_first_ai_exchange_preserving_recent(self, simple_messages):
        """The core bug fix: should remove the OLDEST AI→Tool exchange, keeping recent context."""
        result = remove_up_to_last_ai_message(simple_messages)

        # Should have removed: AI("I'll search...") + Tool("Quantum computing uses qubits...")
        # Remaining: AI("Let me search...") + Tool("Google achieved...") + Human("Please compress...")
        assert len(result) == 3
        assert isinstance(result[0], AIMessage)
        assert "more details" in result[0].content
        assert isinstance(result[1], ToolMessage)
        assert isinstance(result[2], HumanMessage)

    def test_no_ai_messages_returns_original(self, no_ai_messages):
        """When there are no AI messages, the original list should be returned unchanged."""
        result = remove_up_to_last_ai_message(no_ai_messages)
        assert result == no_ai_messages
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        """Empty input should return empty output."""
        result = remove_up_to_last_ai_message([])
        assert result == []

    def test_single_ai_exchange_returns_empty_or_remaining(self, single_ai_exchange):
        """When there's only one AI→Tool exchange, removing it leaves nothing after it."""
        result = remove_up_to_last_ai_message(single_ai_exchange)
        # After removing HumanMessage("Search for X") is before the AI, so we skip AI+Tool
        # Result should be empty since there's nothing after the Tool message
        assert len(result) == 0

    def test_multi_tool_messages_after_ai(self, multi_tool_ai_exchange):
        """AI message followed by multiple tool messages should all be removed together."""
        result = remove_up_to_last_ai_message(multi_tool_ai_exchange)

        # Should remove: AI("I'll run multiple searches") + Tool("Results for A") + Tool("Results for B")
        # Remaining: AI("Here are the final findings")
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert "final findings" in result[0].content

    def test_preserves_most_recent_research(self):
        """After removal, the most recent research (valuable context) should be preserved."""
        messages = [
            HumanMessage(content="topic"),
            AIMessage(content="old search result"),
            ToolMessage(content="old data", name="search", tool_call_id="tc1"),
            AIMessage(content="newer search result"),
            ToolMessage(content="newer data", name="search", tool_call_id="tc2"),
            AIMessage(content="newest search result"),
            ToolMessage(content="newest data", name="search", tool_call_id="tc3"),
        ]
        result = remove_up_to_last_ai_message(messages)

        # Should remove the first AI→Tool pair, keep the rest
        assert len(result) == 4
        assert "newer search result" in result[0].content

    def test_called_twice_removes_two_exchanges(self):
        """Multiple calls should progressively free more context."""
        messages = [
            AIMessage(content="first search"),
            ToolMessage(content="first result", name="search", tool_call_id="tc1"),
            AIMessage(content="second search"),
            ToolMessage(content="second result", name="search", tool_call_id="tc2"),
            AIMessage(content="third search"),
            ToolMessage(content="third result", name="search", tool_call_id="tc3"),
        ]

        # First call removes first exchange
        result = remove_up_to_last_ai_message(messages)
        assert len(result) == 4
        assert "second search" in result[0].content

        # Second call removes next exchange
        result = remove_up_to_last_ai_message(result)
        assert len(result) == 2
        assert "third search" in result[0].content

    def test_ai_message_without_tool_messages(self):
        """An AI message not followed by tool messages should still be removed."""
        messages = [
            AIMessage(content="thinking out loud"),
            HumanMessage(content="continue"),
            AIMessage(content="final response"),
        ]
        result = remove_up_to_last_ai_message(messages)
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)


# ===================================================================
# is_token_limit_exceeded — Provider error detection
# ===================================================================

def _make_exception(class_name: str, module: str, message: str, **attrs):
    """Create an exception instance with a specific class name and module.

    We build a brand-new class (inheriting Exception) whose ``__module__``
    matches what the production helpers inspect at runtime.
    """
    cls = type(class_name, (Exception,), {"__module__": module})
    inst = cls(message)
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


class TestIsTokenLimitExceeded:
    """Tests for detecting token limit exceeded errors across providers."""

    def test_openai_bad_request_with_token_keyword(self):
        """OpenAI BadRequestError with 'token' keyword should be detected."""
        exc = _make_exception(
            "BadRequestError", "openai",
            "maximum context length exceeded with 150000 tokens",
        )
        assert _check_openai_token_limit(exc, str(exc).lower())

    def test_openai_context_length_code(self):
        """OpenAI error with context_length_exceeded code should be detected."""
        exc = Exception("context length exceeded")
        exc.code = "context_length_exceeded"
        exc.type = "invalid_request_error"
        assert _check_openai_token_limit(exc, str(exc).lower())

    def test_anthropic_prompt_too_long(self):
        """Anthropic 'prompt is too long' error should be detected."""
        exc = _make_exception(
            "BadRequestError", "anthropic",
            "prompt is too long: 250000 tokens > 200000 maximum",
        )
        assert _check_anthropic_token_limit(exc, str(exc).lower())

    def test_gemini_resource_exhausted(self):
        """Google ResourceExhausted error should be detected."""
        exc = _make_exception(
            "ResourceExhausted", "google.api_core.exceptions",
            "Resource exhausted",
        )
        assert _check_gemini_token_limit(exc, str(exc).lower())

    def test_unrelated_error_returns_false(self):
        """Non-token-limit errors should not be detected."""
        exc = ValueError("something went wrong")
        assert not is_token_limit_exceeded(exc, "openai:gpt-4.1")

    def test_network_error_returns_false(self):
        """Network errors should not be falsely detected as token limits."""
        exc = ConnectionError("Connection refused")
        assert not is_token_limit_exceeded(exc, "openai:gpt-4.1")

    def test_with_model_name_routes_to_correct_provider(self):
        """Providing a model name should optimize the check to the correct provider."""
        exc = _make_exception(
            "BadRequestError", "anthropic", "prompt is too long",
        )

        # With correct provider prefix, should detect
        assert is_token_limit_exceeded(exc, "anthropic:claude-sonnet-4")
        # With wrong provider prefix, should not detect (only checks that provider)
        assert not is_token_limit_exceeded(exc, "openai:gpt-4.1")

    def test_without_model_name_checks_all_providers(self):
        """Without a model name, should check all providers."""
        exc = _make_exception(
            "BadRequestError", "anthropic", "prompt is too long",
        )
        assert is_token_limit_exceeded(exc)


# ===================================================================
# get_model_token_limit — Model lookup
# ===================================================================

class TestGetModelTokenLimit:
    """Tests for model token limit lookups."""

    def test_known_openai_model(self):
        assert get_model_token_limit("openai:gpt-4.1") == 1047576

    def test_known_anthropic_model(self):
        assert get_model_token_limit("anthropic:claude-sonnet-4") == 200000

    def test_known_google_model(self):
        assert get_model_token_limit("google:gemini-1.5-pro") == 2097152

    def test_known_ollama_model(self):
        assert get_model_token_limit("ollama:mistral") == 32768

    def test_known_bedrock_model(self):
        assert get_model_token_limit("bedrock:us.amazon.nova-premier-v1:0") == 1000000

    def test_unknown_model_returns_none(self):
        assert get_model_token_limit("unknown:mystery-model-7b") is None

    def test_substring_matching(self):
        """Token limit lookup uses substring matching."""
        assert get_model_token_limit("openai:gpt-4.1-mini") == 1047576


# ===================================================================
# get_config_value — Enum/string/dict extraction
# ===================================================================

class TestGetConfigValue:
    """Tests for configuration value extraction."""

    def test_string_value(self):
        assert get_config_value("tavily") == "tavily"

    def test_enum_value(self):
        assert get_config_value(SearchAPI.TAVILY) == "tavily"

    def test_dict_value(self):
        d = {"key": "value"}
        assert get_config_value(d) == d

    def test_none_value(self):
        assert get_config_value(None) is None


# ===================================================================
# get_today_str — Date formatting
# ===================================================================

class TestGetTodayStr:
    """Tests for date formatting utility."""

    def test_returns_non_empty_string(self):
        result = get_today_str()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_year(self):
        result = get_today_str()
        # Should contain a 4-digit year
        import re
        assert re.search(r"\d{4}", result)

    def test_format_matches_expected_pattern(self):
        """Format should be like 'Mon Jan 15, 2024'."""
        import re
        result = get_today_str()
        pattern = r"^[A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2}, \d{4}$"
        assert re.match(pattern, result), f"'{result}' doesn't match expected format"


# ===================================================================
# anthropic_websearch_called — Response parsing
# ===================================================================

class TestAnthropicWebsearchCalled:
    """Tests for detecting Anthropic native web search in responses."""

    def test_detects_web_search(self):
        response = SimpleNamespace(
            response_metadata={
                "usage": {
                    "server_tool_use": {
                        "web_search_requests": 3
                    }
                }
            }
        )
        assert anthropic_websearch_called(response) is True

    def test_no_web_search_zero_requests(self):
        response = SimpleNamespace(
            response_metadata={
                "usage": {
                    "server_tool_use": {
                        "web_search_requests": 0
                    }
                }
            }
        )
        assert anthropic_websearch_called(response) is False

    def test_no_usage_metadata(self):
        response = SimpleNamespace(response_metadata={})
        assert anthropic_websearch_called(response) is False

    def test_no_server_tool_use(self):
        response = SimpleNamespace(
            response_metadata={"usage": {}}
        )
        assert anthropic_websearch_called(response) is False

    def test_malformed_response(self):
        response = SimpleNamespace(response_metadata=None)
        assert anthropic_websearch_called(response) is False


# ===================================================================
# openai_websearch_called — Response parsing
# ===================================================================

class TestOpenAIWebsearchCalled:
    """Tests for detecting OpenAI web search in responses."""

    def test_detects_web_search(self):
        response = SimpleNamespace(
            additional_kwargs={
                "tool_outputs": [
                    {"type": "web_search_call", "content": "search results"}
                ]
            }
        )
        assert openai_websearch_called(response) is True

    def test_no_web_search(self):
        response = SimpleNamespace(
            additional_kwargs={
                "tool_outputs": [
                    {"type": "function_call", "content": "result"}
                ]
            }
        )
        assert openai_websearch_called(response) is False

    def test_no_tool_outputs(self):
        response = SimpleNamespace(additional_kwargs={})
        assert openai_websearch_called(response) is False

    def test_empty_tool_outputs(self):
        response = SimpleNamespace(additional_kwargs={"tool_outputs": []})
        assert openai_websearch_called(response) is False


# ===================================================================
# get_api_key_for_model — Provider API key resolution
# ===================================================================

class TestGetApiKeyForModel:
    """Tests for API key resolution from env vars and config."""

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "OPENAI_API_KEY": "sk-env-openai"})
    def test_openai_from_env(self):
        assert get_api_key_for_model("openai:gpt-4.1", {}) == "sk-env-openai"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "ANTHROPIC_API_KEY": "sk-env-anthropic"})
    def test_anthropic_from_env(self):
        assert get_api_key_for_model("anthropic:claude-sonnet-4", {}) == "sk-env-anthropic"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "GOOGLE_API_KEY": "env-google"})
    def test_google_from_env(self):
        assert get_api_key_for_model("google:gemini-1.5-pro", {}) == "env-google"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "GROQ_API_KEY": "env-groq"})
    def test_groq_from_env(self):
        """Groq provider support (newly added)."""
        assert get_api_key_for_model("groq:llama-3.3-70b", {}) == "env-groq"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "DEEPSEEK_API_KEY": "env-deepseek"})
    def test_deepseek_from_env(self):
        """DeepSeek provider support (newly added)."""
        assert get_api_key_for_model("deepseek:deepseek-chat", {}) == "env-deepseek"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "MISTRAL_API_KEY": "env-mistral"})
    def test_mistral_from_env(self):
        """Mistral provider support (newly added)."""
        assert get_api_key_for_model("mistral:mistral-large", {}) == "env-mistral"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "COHERE_API_KEY": "env-cohere"})
    def test_cohere_from_env(self):
        """Cohere provider support (newly added)."""
        assert get_api_key_for_model("cohere:command-r-plus", {}) == "env-cohere"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "FIREWORKS_API_KEY": "env-fireworks"})
    def test_fireworks_from_env(self):
        """Fireworks provider support (newly added)."""
        assert get_api_key_for_model("fireworks:llama-v3p1-70b", {}) == "env-fireworks"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false"}, clear=False)
    def test_unknown_provider_returns_none(self):
        assert get_api_key_for_model("unknown:model", {}) is None

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "true"})
    def test_openai_from_config(self, mock_runnable_config_with_api_keys):
        result = get_api_key_for_model("openai:gpt-4.1", mock_runnable_config_with_api_keys)
        assert result == "sk-test-openai"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "true"})
    def test_groq_from_config(self, mock_runnable_config_with_api_keys):
        """Groq key from config (newly added provider)."""
        result = get_api_key_for_model("groq:llama-3.3-70b", mock_runnable_config_with_api_keys)
        assert result == "test-groq-key"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "true"})
    def test_config_no_api_keys(self):
        config = {"configurable": {}}
        assert get_api_key_for_model("openai:gpt-4.1", config) is None

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false"}, clear=False)
    def test_case_insensitive_model_name(self):
        """Model name matching should be case-insensitive."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            assert get_api_key_for_model("OpenAI:GPT-4.1", {}) == "sk-test"

    @patch.dict(os.environ, {"GET_API_KEYS_FROM_CONFIG": "false", "AWS_ACCESS_KEY_ID": "env-aws"})
    def test_bedrock_from_env(self):
        """AWS Bedrock provider support (newly added)."""
        assert get_api_key_for_model("bedrock:us.anthropic.claude-sonnet-4", {}) == "env-aws"


# ===================================================================
# get_notes_from_tool_calls — Tool message extraction
# ===================================================================

class TestGetNotesFromToolCalls:
    """Tests for extracting notes from tool call messages."""

    def test_extracts_tool_message_content(self):
        messages = [
            HumanMessage(content="topic"),
            AIMessage(content="searching"),
            ToolMessage(content="Result 1", name="search", tool_call_id="tc1"),
            ToolMessage(content="Result 2", name="search", tool_call_id="tc2"),
        ]
        notes = get_notes_from_tool_calls(messages)
        assert notes == ["Result 1", "Result 2"]

    def test_no_tool_messages(self):
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="world"),
        ]
        notes = get_notes_from_tool_calls(messages)
        assert notes == []

    def test_empty_messages(self):
        notes = get_notes_from_tool_calls([])
        assert notes == []
