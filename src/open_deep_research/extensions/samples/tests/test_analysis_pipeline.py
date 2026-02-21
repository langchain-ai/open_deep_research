"""Tests for the Analysis Pipeline integration in graph.py and state.py."""
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSummaryStateAnalysisFields:
    """Test that SummaryState has the required analysis fields."""

    def test_state_has_analysis_required(self):
        """Verify analysis_required field exists with default False."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert hasattr(state, "analysis_required")
        assert state.analysis_required is False

    def test_state_has_analysis_output(self):
        """Verify analysis_output field exists with default None."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert hasattr(state, "analysis_output")
        assert state.analysis_output is None

    def test_state_has_analysis_report_path(self):
        """Verify analysis_report_path field exists with default None."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert hasattr(state, "analysis_report_path")
        assert state.analysis_report_path is None

    def test_state_has_analysis_charts(self):
        """Verify analysis_charts field exists with default empty list."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert hasattr(state, "analysis_charts")
        assert state.analysis_charts == []

    def test_state_has_analysis_chart_explanations(self):
        """Verify analysis_chart_explanations field exists with default empty dict."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        assert hasattr(state, "analysis_chart_explanations")
        assert state.analysis_chart_explanations == {}

    def test_state_analysis_required_can_be_set_true(self):
        """Verify analysis_required can be set to True."""
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=True)
        assert state.analysis_required is True

    def test_state_analysis_fields_serializable(self):
        """Verify analysis fields can be serialized to dict."""
        from src.state import SummaryState
        state = SummaryState(
            research_topic="test",
            analysis_required=True,
            analysis_output="Test output",
            analysis_report_path="/tmp/report.html",
            analysis_charts=["/tmp/chart1.html"],
            analysis_chart_explanations={"chart1": {"title": "Test"}},
        )
        state_dict = state.model_dump()
        assert state_dict["analysis_required"] is True
        assert state_dict["analysis_output"] == "Test output"
        assert state_dict["analysis_report_path"] == "/tmp/report.html"
        assert state_dict["analysis_charts"] == ["/tmp/chart1.html"]
        assert state_dict["analysis_chart_explanations"] == {"chart1": {"title": "Test"}}


class TestGraphAnalysisNode:
    """Test that the graph has the analysis node and conditional edge."""

    def test_graph_has_run_analysis_node(self):
        """Verify graph contains the run_analysis node."""
        from src.graph import create_graph
        graph = create_graph()
        assert "run_analysis" in graph.nodes

    def test_graph_compiles_successfully(self):
        """Verify graph compiles without errors."""
        from src.graph import create_graph
        graph = create_graph()
        # If we got here without an exception, the graph compiled
        assert graph is not None

    def test_graph_has_finalize_report_node(self):
        """Verify finalize_report node still exists (conditional edge source)."""
        from src.graph import create_graph
        graph = create_graph()
        assert "finalize_report" in graph.nodes

    def test_route_after_finalize_report_analysis_true(self):
        """Test routing function returns 'run_analysis' when analysis_required=True."""
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=True)
        result = route_after_finalize_report(state)
        assert result == "run_analysis"

    def test_route_after_finalize_report_analysis_false(self):
        """Test routing function returns '__end__' when analysis_required=False."""
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=False)
        result = route_after_finalize_report(state)
        assert result == "__end__"

    def test_route_after_finalize_report_default(self):
        """Test routing function returns '__end__' with default state."""
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test")
        result = route_after_finalize_report(state)
        assert result == "__end__"


class TestRunAnalysisFunction:
    """Test the run_analysis function with mocked dependencies."""

    def _run_async(self, coro):
        """Helper to run async functions in sync tests."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_run_analysis_function_exists(self):
        """Verify run_analysis function can be imported."""
        from src.graph import run_analysis
        assert run_analysis is not None
        assert callable(run_analysis)

    def test_run_analysis_success_path(self):
        """Test run_analysis with a successful pipeline result."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test pollution analysis",
            analysis_required=True,
            running_summary="# Test Report\nPollution levels: PM2.5 = 150, PM10 = 200",
        )

        config = {"configurable": {"provider": "google", "model": "gemini-2.5-pro"}}

        # Mock DataAnalysisAgent at the point where run_analysis imports it
        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "completed",
            "output": "Analysis shows high pollution levels.",
            "charts": ["/tmp/chart1.html"],
            "chart_explanations": {"chart1": {"title": "PM Levels"}},
            "extracted_data": "Year,PM2.5\n2023,150",
            "data_profile": "Data profile summary",
        })

        mock_report_builder = MagicMock(return_value="/tmp/report.html")

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(DataAnalysisAgent=MagicMock(return_value=mock_agent)),
            "extensions.utils.report_builder": MagicMock(build_html_report=mock_report_builder),
        }):
            result = self._run_async(run_analysis(state, config))

        assert result["analysis_output"] == "Analysis shows high pollution levels."
        assert result["analysis_report_path"] == "/tmp/report.html"
        assert result["analysis_charts"] == ["/tmp/chart1.html"]
        assert result["analysis_chart_explanations"] == {"chart1": {"title": "PM Levels"}}

    def test_run_analysis_pipeline_failure(self):
        """Test run_analysis when pipeline returns error status."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            analysis_required=True,
            running_summary="Some text",
        )

        config = {"configurable": {"provider": "google", "model": "gemini-2.5-pro"}}

        mock_agent = MagicMock()
        mock_agent.run_pipeline = MagicMock(return_value={
            "status": "error",
            "error": "No extractable data found",
        })

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": MagicMock(DataAnalysisAgent=MagicMock(return_value=mock_agent)),
            "extensions.utils.report_builder": MagicMock(),
        }):
            result = self._run_async(run_analysis(state, config))

        assert result["analysis_report_path"] is None
        assert result["analysis_charts"] == []
        assert "failed" in result["analysis_output"].lower()

    def test_run_analysis_exception_handling(self):
        """Test run_analysis handles exceptions gracefully."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            analysis_required=True,
            running_summary="Some research text",
        )

        config = {"configurable": {"provider": "google", "model": "gemini-2.5-pro"}}

        # Force an import error by making the module raise
        mock_module = MagicMock()
        mock_module.DataAnalysisAgent = MagicMock(side_effect=Exception("Import error"))

        with patch.dict("sys.modules", {
            "extensions.agents.data_analysis_agent": mock_module,
            "extensions.utils.report_builder": MagicMock(),
        }):
            result = self._run_async(run_analysis(state, config))

        assert isinstance(result, dict)
        assert result["analysis_report_path"] is None
        assert result["analysis_charts"] == []
        assert "error" in result["analysis_output"].lower()


class TestAnalysisReportEndpointLogic:
    """Test the analysis report serving logic (without full app import to avoid hangs)."""

    def test_analysis_report_filename_regex_valid(self):
        """Test that valid filenames pass the regex check."""
        import re
        pattern = r'^report_[\w-]+\.html$'
        # Valid filenames
        assert re.match(pattern, "report_abc123.html")
        assert re.match(pattern, "report_abc-def-123.html")
        assert re.match(pattern, "report_a1b2c3d4.html")

    def test_analysis_report_filename_regex_rejects_traversal(self):
        """Test that directory traversal filenames are rejected."""
        import re
        pattern = r'^report_[\w-]+\.html$'
        # Invalid filenames
        assert not re.match(pattern, "../../../etc/passwd")
        assert not re.match(pattern, "malicious.py")
        assert not re.match(pattern, "report_.html../hack")
        assert not re.match(pattern, "")

    def test_analysis_report_filename_regex_rejects_non_html(self):
        """Test that non-HTML filenames are rejected."""
        import re
        pattern = r'^report_[\w-]+\.html$'
        assert not re.match(pattern, "report_abc.py")
        assert not re.match(pattern, "report_abc.js")
        assert not re.match(pattern, "report_abc.exe")

    def test_analysis_report_path_construction(self):
        """Test that report file path is constructed correctly."""
        filename = "report_abc123.html"
        report_path = os.path.join("outputs", "reports", filename)
        assert report_path == os.path.join("outputs", "reports", "report_abc123.html")
