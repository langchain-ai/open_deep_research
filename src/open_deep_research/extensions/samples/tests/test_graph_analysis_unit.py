"""Unit tests for graph.py - run_analysis and route_after_finalize_report.

All external dependencies (DataAnalysisAgent, report_builder) are mocked.
"""
import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRouteAfterFinalizeReport:
    """Test the routing function for analysis."""

    def test_routes_to_analysis_when_true(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=True)
        assert route_after_finalize_report(state) == "run_analysis"

    def test_routes_to_end_when_false(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=False)
        assert route_after_finalize_report(state) == "__end__"

    def test_routes_to_end_by_default(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert route_after_finalize_report(state) == "__end__"


class TestRunAnalysis:
    """Test run_analysis node function with mocked dependencies."""

    def _make_state(self, **kwargs):
        from src.state import SummaryState
        defaults = {
            "research_topic": "test topic",
            "analysis_required": True,
            "markdown_report": "# Report\nGDP is 25.5 trillion.",
            "running_summary": "<html>Fallback</html>",
        }
        defaults.update(kwargs)
        return SummaryState(**defaults)

    def _make_config(self, provider="azure", model="gpt-4o"):
        return {"configurable": {"provider": provider, "model": model}}

    def test_success_path(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Analysis done.",
            "charts": ["/tmp/chart1.html"],
            "chart_explanations": {"/tmp/chart1.html": {"title": "T", "explanation": "E"}},
            "extracted_data": "A,B\n1,2",
            "data_profile": "Profile data",
        })

        mock_report_builder = MagicMock(return_value="/tmp/report.html")

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=mock_report_builder
            ),
        }):
            result = _run_async(run_analysis(state, config))

        assert result["analysis_output"] == "Analysis done."
        assert result["analysis_report_path"] == "/tmp/report.html"
        assert result["analysis_charts"] == ["/tmp/chart1.html"]

    def test_prefers_markdown_report_over_running_summary(self):
        from src.graph import run_analysis

        state = self._make_state(
            markdown_report="Clean markdown content",
            running_summary="<div>HTML content</div>",
        )
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Done",
            "charts": [],
            "chart_explanations": {},
            "extracted_data": "",
            "data_profile": "",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=MagicMock(return_value="/tmp/r.html")
            ),
        }):
            result = _run_async(run_analysis(state, config))

        # Verify run_pipeline was called with markdown_report, not running_summary
        pipeline_arg = mock_agent.run_pipeline.call_args[0][0]
        assert pipeline_arg == "Clean markdown content"

    def test_falls_back_to_running_summary(self):
        from src.graph import run_analysis

        state = self._make_state(
            markdown_report="",  # Empty markdown
            running_summary="HTML fallback content",
        )
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Done",
            "charts": [],
            "chart_explanations": {},
            "extracted_data": "",
            "data_profile": "",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=MagicMock(return_value="/tmp/r.html")
            ),
        }):
            result = _run_async(run_analysis(state, config))

        pipeline_arg = mock_agent.run_pipeline.call_args[0][0]
        assert pipeline_arg == "HTML fallback content"

    def test_pipeline_error_status(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "error",
            "error": "No data found",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(),
        }):
            result = _run_async(run_analysis(state, config))

        assert result["analysis_report_path"] is None
        assert result["analysis_charts"] == []
        assert "failed" in result["analysis_output"].lower()

    def test_exception_handling(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config()

        mock_module = MagicMock()
        mock_module.DataAnalysisAgent = MagicMock(side_effect=RuntimeError("Init failed"))

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": mock_module,
            "extensions.utils.report_builder": MagicMock(),
        }):
            result = _run_async(run_analysis(state, config))

        assert result["analysis_report_path"] is None
        assert result["analysis_charts"] == []
        assert "error" in result["analysis_output"].lower()

    def test_report_builder_failure(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Done",
            "charts": [],
            "chart_explanations": {},
            "extracted_data": "",
            "data_profile": "",
        })

        mock_report_builder = MagicMock(side_effect=Exception("Template error"))

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=mock_report_builder
            ),
        }):
            result = _run_async(run_analysis(state, config))

        # Pipeline succeeded but report failed
        assert result["analysis_output"] == "Done"
        assert result["analysis_report_path"] is None

    def test_returns_provider_model(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config(provider="gemini", model="gemini-2.5-flash")

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Done",
            "charts": [],
            "chart_explanations": {},
            "extracted_data": "",
            "data_profile": "",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=MagicMock(return_value="/tmp/r.html")
            ),
        }):
            result = _run_async(run_analysis(state, config))

        assert result["analysis_required"] is True

    def test_result_has_all_expected_keys(self):
        from src.graph import run_analysis

        state = self._make_state()
        config = self._make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Done",
            "charts": [],
            "chart_explanations": {},
            "extracted_data": "",
            "data_profile": "",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(
                DataAnalysisAgent=MagicMock(return_value=mock_agent)
            ),
            "extensions.utils.report_builder": MagicMock(
                build_html_report=MagicMock(return_value="/tmp/r.html")
            ),
        }):
            result = _run_async(run_analysis(state, config))

        expected_keys = {
            "analysis_output", "analysis_report_path",
            "analysis_charts", "analysis_chart_explanations",
            "analysis_required", "llm_provider", "llm_model",
        }
        assert expected_keys.issubset(set(result.keys()))
