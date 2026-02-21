"""Tests for the Intent Detection Agent (services/intent_detector.py)."""
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestDetectIntentParsing:
    """Test the intent detection parsing logic without requiring LLM calls."""

    def test_module_imports(self):
        """Verify intent_detector module can be imported."""
        from services.intent_detector import detect_intent, INTENT_DETECTION_PROMPT
        assert detect_intent is not None
        assert INTENT_DETECTION_PROMPT is not None

    def test_prompt_contains_required_elements(self):
        """Verify the intent detection prompt has correct structure."""
        from services.intent_detector import INTENT_DETECTION_PROMPT
        # Should describe two categories
        assert "Research only" in INTENT_DETECTION_PROMPT
        assert "Research + Analysis" in INTENT_DETECTION_PROMPT
        # Should specify JSON output format
        assert "analysis_required" in INTENT_DETECTION_PROMPT
        assert "true" in INTENT_DETECTION_PROMPT
        assert "false" in INTENT_DETECTION_PROMPT

    def _run_async(self, coro):
        """Helper to run async functions in sync tests."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_detect_intent_research_only(self):
        """Test that a pure research query returns analysis_required=False."""
        mock_response = MagicMock()
        mock_response.content = '{"analysis_required": false}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(
                detect_intent("What is the history of the Roman Empire?")
            )

        assert result["analysis_required"] is False

    def test_detect_intent_analysis_required(self):
        """Test that an analysis query returns analysis_required=True."""
        mock_response = MagicMock()
        mock_response.content = '{"analysis_required": true}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(
                detect_intent(
                    "Research air pollution in Delhi and generate charts showing pollution levels over time"
                )
            )

        assert result["analysis_required"] is True

    def test_detect_intent_handles_markdown_code_block(self):
        """Test parsing when LLM wraps JSON in markdown code block."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"analysis_required": true}\n```'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(
                detect_intent("Analyze market trends with visualizations")
            )

        assert result["analysis_required"] is True

    def test_detect_intent_handles_plain_code_block(self):
        """Test parsing when LLM wraps JSON in plain code block."""
        mock_response = MagicMock()
        mock_response.content = '```\n{"analysis_required": false}\n```'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(
                detect_intent("Research quantum computing")
            )

        assert result["analysis_required"] is False

    def test_detect_intent_llm_error_defaults_false(self):
        """Test that LLM errors default to analysis_required=False (safe fallback)."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(detect_intent("Some query"))

        assert result["analysis_required"] is False

    def test_detect_intent_invalid_json_defaults_false(self):
        """Test that invalid JSON response defaults to analysis_required=False."""
        mock_response = MagicMock()
        mock_response.content = "I think this needs analysis"  # Not JSON

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(detect_intent("Some query"))

        assert result["analysis_required"] is False

    def test_detect_intent_uses_provided_provider_model(self):
        """Test that provider and model are correctly passed to get_llm_client."""
        mock_response = MagicMock()
        mock_response.content = '{"analysis_required": false}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm) as mock_get_llm:
            from services.intent_detector import detect_intent
            self._run_async(
                detect_intent("Test query", provider="google", model="gemini-2.5-pro")
            )

            mock_get_llm.assert_called_once_with("google", "gemini-2.5-pro")

    def test_detect_intent_defaults_provider_from_env(self):
        """Test that provider defaults to env var when not provided."""
        mock_response = MagicMock()
        mock_response.content = '{"analysis_required": false}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o"}):
            with patch("llm_clients.get_llm_client", return_value=mock_llm) as mock_get_llm:
                from services.intent_detector import detect_intent
                self._run_async(detect_intent("Test query"))

                mock_get_llm.assert_called_once_with("openai", "gpt-4o")

    def test_detect_intent_return_structure(self):
        """Test that detect_intent always returns a dict with 'analysis_required' key."""
        mock_response = MagicMock()
        mock_response.content = '{"analysis_required": true}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            from services.intent_detector import detect_intent
            result = self._run_async(detect_intent("Test query"))

        assert isinstance(result, dict)
        assert "analysis_required" in result
        assert isinstance(result["analysis_required"], bool)
