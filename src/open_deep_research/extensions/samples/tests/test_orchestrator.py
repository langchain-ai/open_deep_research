"""
Unit tests for orchestrator.py
==============================
Run with:  python -m pytest tests/test_orchestrator.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import (
    RESEARCH_TYPES,
    MODEL_PROVIDER_MAP,
    build_config,
    detect_provider,
    parse_args,
    sanitize_folder_name,
    save_output,
    execute_research,
)


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------

class TestDetectProvider:
    def test_ollama_models(self):
        assert detect_provider("mistral") == "ollama"
        assert detect_provider("deepseek-r1") == "ollama"
        assert detect_provider("llama3") == "ollama"

    def test_azure_model(self):
        assert detect_provider("gpt-4o") == "azure"

    def test_openai_models(self):
        assert detect_provider("o3-mini") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"

    def test_anthropic_models(self):
        assert detect_provider("claude-sonnet-4-5-20250929") == "anthropic"

    def test_google_models(self):
        assert detect_provider("gemini-2.5-pro") == "google"

    @patch.dict(os.environ, {"LLM_PROVIDER": "groq"})
    def test_unknown_model_falls_back_to_env(self):
        assert detect_provider("some-unknown-model") == "groq"

    @patch.dict(os.environ, {}, clear=True)
    def test_unknown_model_no_env_defaults_openai(self):
        # Remove LLM_PROVIDER if present
        os.environ.pop("LLM_PROVIDER", None)
        assert detect_provider("some-unknown-model") == "openai"


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_basic_config(self):
        config = build_config("ollama", "mistral", 3)
        assert config["configurable"]["llm_provider"] == "ollama"
        assert config["configurable"]["llm_model"] == "mistral"
        assert config["configurable"]["max_web_research_loops"] == 3
        assert config["recursion_limit"] == 100

    def test_search_api_from_env(self):
        with patch.dict(os.environ, {"SEARCH_API": "tavily"}):
            config = build_config("openai", "gpt-4o", 10)
            assert config["configurable"]["search_api"] == "tavily"

    def test_different_loop_counts(self):
        for rtype, loops in RESEARCH_TYPES.items():
            config = build_config("ollama", "mistral", loops)
            assert config["configurable"]["max_web_research_loops"] == loops


# ---------------------------------------------------------------------------
# RESEARCH_TYPES mapping
# ---------------------------------------------------------------------------

class TestResearchTypes:
    def test_quick(self):
        assert RESEARCH_TYPES["quick"] == 1

    def test_standard(self):
        assert RESEARCH_TYPES["standard"] == 3

    def test_deep(self):
        assert RESEARCH_TYPES["deep"] == 10

    def test_all_types_present(self):
        assert set(RESEARCH_TYPES.keys()) == {"quick", "standard", "deep"}


# ---------------------------------------------------------------------------
# sanitize_folder_name
# ---------------------------------------------------------------------------

class TestSanitizeFolderName:
    def test_basic(self):
        assert sanitize_folder_name("Hello World") == "hello_world"

    def test_special_chars(self):
        result = sanitize_folder_name("What is AI? (2025)")
        assert "?" not in result
        assert "(" not in result

    def test_max_length(self):
        long_query = "a" * 100
        result = sanitize_folder_name(long_query, max_len=50)
        assert len(result) <= 50

    def test_empty_string(self):
        result = sanitize_folder_name("")
        assert result == ""

    def test_unicode(self):
        result = sanitize_folder_name("quantum computing & AI")
        assert "&" not in result


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_query(self):
        args = parse_args(["--query", "test topic"])
        assert args.query == "test topic"

    def test_missing_query_raises(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_default_research_type(self):
        args = parse_args(["--query", "test"])
        assert args.research_type == "standard"

    def test_quick_type(self):
        args = parse_args(["--query", "test", "--type", "quick"])
        assert args.research_type == "quick"

    def test_deep_type(self):
        args = parse_args(["--query", "test", "--type", "deep"])
        assert args.research_type == "deep"

    def test_invalid_type_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--query", "test", "--type", "invalid"])

    def test_no_stream_flag(self):
        args = parse_args(["--query", "test", "--no-stream"])
        assert args.no_stream is True

    def test_stream_default(self):
        args = parse_args(["--query", "test"])
        assert args.no_stream is False

    def test_provider_override(self):
        args = parse_args(["--query", "test", "--provider", "azure"])
        assert args.provider == "azure"

    def test_model_override(self):
        args = parse_args(["--query", "test", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_all_args_together(self):
        args = parse_args([
            "--query", "AI safety",
            "--model", "gpt-4o",
            "--provider", "azure",
            "--type", "deep",
            "--no-stream",
        ])
        assert args.query == "AI safety"
        assert args.model == "gpt-4o"
        assert args.provider == "azure"
        assert args.research_type == "deep"
        assert args.no_stream is True


# ---------------------------------------------------------------------------
# save_output
# ---------------------------------------------------------------------------

class TestSaveOutput:
    def _mock_result(self):
        return {
            "running_summary": "# Test Report\n\nThis is a test research report.",
            "sources_gathered": ["Source 1 : https://example.com"],
            "source_citations": {"1": {"title": "Source 1", "url": "https://example.com"}},
            "research_loop_count": 2,
            "research_complete": True,
        }

    def test_creates_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = self._mock_result()
                output_dir = save_output(result, "test query", "ollama", "mistral", 3, 10.5, False)
                assert output_dir.exists()
            finally:
                os.chdir(original_cwd)

    def test_creates_report_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = self._mock_result()
                output_dir = save_output(result, "test", "ollama", "mistral", 3, 5.0, True)
                report_path = output_dir / "report.md"
                assert report_path.exists()
                content = report_path.read_text(encoding="utf-8")
                assert "Test Report" in content
            finally:
                os.chdir(original_cwd)

    def test_creates_sources_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = self._mock_result()
                output_dir = save_output(result, "test", "ollama", "mistral", 3, 5.0, False)
                sources_path = output_dir / "sources.json"
                assert sources_path.exists()
                data = json.loads(sources_path.read_text(encoding="utf-8"))
                assert "sources_gathered" in data
                assert "source_citations" in data
                assert len(data["sources_gathered"]) == 1
            finally:
                os.chdir(original_cwd)

    def test_creates_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = self._mock_result()
                output_dir = save_output(result, "my query", "azure", "gpt-4o", 10, 120.0, True)
                meta_path = output_dir / "metadata.json"
                assert meta_path.exists()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert meta["query"] == "my query"
                assert meta["provider"] == "azure"
                assert meta["model"] == "gpt-4o"
                assert meta["research_type_max_loops"] == 10
                assert meta["streaming"] is True
                assert meta["duration_seconds"] == 120.0
            finally:
                os.chdir(original_cwd)

    def test_empty_result_still_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = {}
                output_dir = save_output(result, "empty", "ollama", "mistral", 1, 1.0, False)
                assert (output_dir / "report.md").exists()
                assert (output_dir / "sources.json").exists()
                assert (output_dir / "metadata.json").exists()
                # Report should have fallback text
                report = (output_dir / "report.md").read_text(encoding="utf-8")
                assert report == "No report generated."
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# execute_research (with mocked graph)
# ---------------------------------------------------------------------------

class TestExecuteResearch:
    def _mock_graph(self, invoke_result=None):
        graph = MagicMock()
        graph.invoke.return_value = invoke_result or {
            "running_summary": "Mock report",
            "sources_gathered": [],
            "source_citations": {},
            "research_loop_count": 1,
            "research_complete": True,
        }
        return graph

    def test_non_streaming_calls_invoke(self):
        graph = self._mock_graph()
        config = build_config("ollama", "mistral", 1)
        result = execute_research(graph, "test query", config, stream=False)
        graph.invoke.assert_called_once()
        assert result["running_summary"] == "Mock report"

    def test_streaming_fallback_on_error(self):
        """If streaming fails, it should fallback to graph.invoke()."""
        graph = self._mock_graph()
        # Make astream_events raise an error
        graph.astream_events = MagicMock(side_effect=Exception("streaming not supported"))
        config = build_config("ollama", "mistral", 1)
        result = execute_research(graph, "test query", config, stream=True)
        # Should have fallen back to invoke
        graph.invoke.assert_called_once()
        assert result["running_summary"] == "Mock report"

    def test_non_streaming_returns_result(self):
        custom_result = {
            "running_summary": "Custom report content",
            "sources_gathered": ["src1", "src2"],
            "source_citations": {"1": {"title": "src1", "url": "http://a.com"}},
            "research_loop_count": 3,
            "research_complete": True,
        }
        graph = self._mock_graph(custom_result)
        config = build_config("azure", "gpt-4o", 3)
        result = execute_research(graph, "AI trends", config, stream=False)
        assert result["running_summary"] == "Custom report content"
        assert len(result["sources_gathered"]) == 2
        assert result["research_loop_count"] == 3


# ---------------------------------------------------------------------------
# MODEL_PROVIDER_MAP completeness
# ---------------------------------------------------------------------------

class TestModelProviderMap:
    def test_all_values_are_valid_providers(self):
        valid_providers = {"ollama", "azure", "openai", "anthropic", "google", "groq"}
        for model, provider in MODEL_PROVIDER_MAP.items():
            assert provider in valid_providers, f"Model {model} maps to invalid provider {provider}"

    def test_map_is_not_empty(self):
        assert len(MODEL_PROVIDER_MAP) > 0
