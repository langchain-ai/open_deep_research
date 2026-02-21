"""
Tests for:
1. analysis_required propagation through finalize_report / reflect_on_report
2. run_analysis node invocation when analysis_required=True
3. Research report generation via finalize_report
"""
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure project root and src are on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine in a fresh event loop (test helper)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(provider="gemini", model="gemini-2.5-flash"):
    return {"configurable": {"provider": provider, "model": model}}


# ---------------------------------------------------------------------------
# 1. analysis_required PROPAGATION
#    Bug: finalize_report / reflect_on_report used to drop analysis_required
#    from their return dicts, causing the routing to always go to END.
# ---------------------------------------------------------------------------

class TestAnalysisRequiredPropagation:
    """Verify that finalize_report and reflect_on_report preserve analysis_required."""

    # --- finalize_report ---

    def test_finalize_report_preserves_analysis_required_true(self):
        """finalize_report must include analysis_required=True in its return dict."""
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="Delhi pollution",
            running_summary="## Summary\nPM2.5 is 150.",
            analysis_required=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Final Report\nDelhi pollution summary.")

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            result = finalize_report(state, config)

        assert isinstance(result, dict), "finalize_report must return a dict"
        assert "analysis_required" in result, (
            "analysis_required MISSING from finalize_report return — routing will break"
        )
        assert result["analysis_required"] is True, (
            f"Expected True, got {result['analysis_required']!r} — "
            "run_analysis will never be called"
        )

    def test_finalize_report_preserves_analysis_required_false(self):
        """finalize_report must include analysis_required=False when not set."""
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            running_summary="## Summary\nSome content.",
            analysis_required=False,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Final Report\nContent.")

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            result = finalize_report(state, config)

        assert result.get("analysis_required") is False

    def test_finalize_report_preserves_llm_provider(self):
        """finalize_report must pass through llm_provider so run_analysis uses the right LLM."""
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            running_summary="## Summary\nContent.",
            analysis_required=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Report\nContent.")

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            result = finalize_report(state, config)

        assert result.get("llm_provider") == "gemini"
        assert result.get("llm_model") == "gemini-2.5-flash"

    # --- reflect_on_report ---

    def test_reflect_on_report_preserves_analysis_required_true(self):
        """reflect_on_report must not drop analysis_required=True from its return dict."""
        from src.graph import reflect_on_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="Delhi pollution",
            running_summary="# Report\nPM2.5 = 150.",
            analysis_required=True,
            research_loop_count=0,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            extra_effort=False,
            minimum_effort=False,
        )
        config = _make_config()

        # Simulate LLM returning a decision to stop research
        mock_response = MagicMock(content="""
REFLECTION:
The research covers the main pollution metrics adequately.

KNOWLEDGE_GAP: None. The summary is comprehensive.

SEARCH_QUERY: (none needed)

RESEARCH_COMPLETE: True

PRIORITY_SECTION: Air Quality Index
""")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        # reflect_on_report does an inline `from llm_clients import get_llm_client`
        with patch("llm_clients.get_llm_client", return_value=mock_llm):
            result = reflect_on_report(state, config)

        assert "analysis_required" in result, (
            "reflect_on_report must include analysis_required in its return dict"
        )
        assert result["analysis_required"] is True


# ---------------------------------------------------------------------------
# 2. ROUTING: route_after_finalize_report
# ---------------------------------------------------------------------------

class TestRouteAfterFinalizeReport:
    """Test the conditional edge routing function."""

    def test_routes_to_run_analysis_when_true(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=True)
        assert route_after_finalize_report(state) == "run_analysis"

    def test_routes_to_end_when_false(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test", analysis_required=False)
        result = route_after_finalize_report(state)
        assert result in ("__end__", "END", None) or "end" in str(result).lower()

    def test_routes_to_end_by_default(self):
        from src.graph import route_after_finalize_report
        from src.state import SummaryState
        state = SummaryState(research_topic="test")  # default analysis_required=False
        result = route_after_finalize_report(state)
        assert result not in ("run_analysis",), (
            "With analysis_required=False (default), should NOT route to run_analysis"
        )


# ---------------------------------------------------------------------------
# 3. run_analysis INVOCATION
# ---------------------------------------------------------------------------

class TestRunAnalysisInvocation:
    """Test that run_analysis calls DataAnalysisAgent with correct provider/model."""

    def test_run_analysis_calls_agent_with_provider_and_model(self):
        """run_analysis must pass the resolved provider and model to DataAnalysisAgent."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="Delhi pollution analysis",
            running_summary="## Data\nPM2.5 levels: 150 µg/m³ in 2024.",
            analysis_required=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config(provider="gemini", model="gemini-2.5-flash")

        mock_agent = MagicMock()
        mock_agent.run_pipeline.return_value = {
            "status": "completed",
            "output": "High pollution levels detected.",
            "charts": ["/tmp/chart1.html"],
            "chart_explanations": {"chart1": {"title": "PM2.5 Trend"}},
            "extracted_data": "Year,PM2.5\n2024,150",
            "data_profile": "25 rows, 2 columns",
        }

        MockAgentClass = MagicMock(return_value=mock_agent)
        mock_build_report = MagicMock(return_value="/tmp/analysis_report.html")

        # run_analysis does inline imports: DataAnalysisAgent and build_html_report
        with patch("extensions.agents.data_analysis_agent.DataAnalysisAgent", MockAgentClass), \
             patch("extensions.utils.report_builder.build_html_report", mock_build_report):
            result = _run(run_analysis(state, config))

        # Verify DataAnalysisAgent was called with provider and model
        MockAgentClass.assert_called_once()
        call_kwargs = MockAgentClass.call_args
        # Accept both positional and keyword args
        args, kwargs = call_kwargs
        all_kwargs = {**kwargs}
        if len(args) >= 1:
            all_kwargs.setdefault("provider", args[0])
        if len(args) >= 2:
            all_kwargs.setdefault("model", args[1])

        assert all_kwargs.get("provider") == "gemini", (
            f"DataAnalysisAgent should be called with provider='gemini', got: {all_kwargs}"
        )
        assert all_kwargs.get("model") == "gemini-2.5-flash", (
            f"DataAnalysisAgent should be called with model='gemini-2.5-flash', got: {all_kwargs}"
        )

    def test_run_analysis_returns_analysis_output(self):
        """run_analysis must return analysis_output in its result dict."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="pollution test",
            running_summary="PM2.5 = 200 in Jan 2026.",
            analysis_required=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()

        mock_agent = MagicMock()
        mock_agent.run_pipeline.return_value = {
            "status": "completed",
            "output": "Pollution analysis complete. Peak PM2.5 in winter.",
            "charts": [],
            "chart_explanations": {},
        }
        MockAgentClass = MagicMock(return_value=mock_agent)
        mock_build_report = MagicMock(return_value="/tmp/analysis_report.html")

        with patch("extensions.agents.data_analysis_agent.DataAnalysisAgent", MockAgentClass), \
             patch("extensions.utils.report_builder.build_html_report", mock_build_report):
            result = _run(run_analysis(state, config))

        assert "analysis_output" in result
        assert result["analysis_output"] == "Pollution analysis complete. Peak PM2.5 in winter."

    def test_run_analysis_handles_agent_exception(self):
        """run_analysis must not crash when DataAnalysisAgent raises an exception."""
        from src.graph import run_analysis
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            running_summary="some text",
            analysis_required=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()

        MockAgentClass = MagicMock(side_effect=RuntimeError("LLM connection failed"))

        with patch("extensions.agents.data_analysis_agent.DataAnalysisAgent", MockAgentClass):
            result = _run(run_analysis(state, config))

        assert isinstance(result, dict), "run_analysis must always return a dict even on error"
        assert result.get("analysis_charts") == []
        assert result.get("analysis_report_path") is None


# ---------------------------------------------------------------------------
# 4. REPORT GENERATION — finalize_report output quality
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Test that finalize_report produces a properly structured report."""

    def _call_finalize(self, running_summary, llm_content="# Report\n\n## Section\nContent here."):
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="Delhi pollution",
            running_summary=running_summary,
            analysis_required=False,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            source_citations={
                "1": {"title": "NDTV AQI", "url": "https://ndtv.com/aqi/delhi"},
                "2": {"title": "IQAir Delhi", "url": "https://iqair.com/india/delhi"},
            },
        )
        config = _make_config()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_content)

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            return finalize_report(state, config)

    def test_report_returns_running_summary(self):
        """finalize_report must return a 'running_summary' key with the report text."""
        result = self._call_finalize(
            running_summary="## Delhi Pollution\nPM2.5 averages 150 µg/m³.",
            llm_content="# Delhi Pollution Report\n\n## Overview\nHigh PM2.5 levels."
        )
        assert "running_summary" in result
        assert isinstance(result["running_summary"], str)
        assert len(result["running_summary"]) > 0

    def test_report_running_summary_not_empty(self):
        """The finalized report must not be an empty string."""
        result = self._call_finalize(
            running_summary="## Summary\nAQI in Delhi reached 300 in Jan 2026.",
            llm_content="# Delhi Air Quality 2026\n\n## Key Findings\nAQI = 300."
        )
        assert result["running_summary"].strip() != ""

    def test_report_preserves_source_citations(self):
        """finalize_report must carry source_citations forward."""
        result = self._call_finalize(
            running_summary="## Delhi Pollution\nSome content.",
        )
        assert "source_citations" in result
        assert len(result["source_citations"]) >= 2

    def test_report_includes_markdown_version(self):
        """finalize_report must also return a markdown_report key."""
        result = self._call_finalize(
            running_summary="## Summary\nContent.",
            llm_content="# Report\n\n## Section\nContent."
        )
        assert "markdown_report" in result
        assert isinstance(result["markdown_report"], str)

    def test_report_preserves_visualizations(self):
        """finalize_report must not drop the visualizations list."""
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            running_summary="## Summary\nContent.",
            analysis_required=False,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            visualizations=[{"filename": "chart1.html", "title": "PM2.5 Trend"}],
        )
        config = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Report\nContent.")

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            result = finalize_report(state, config)

        assert result.get("visualizations") == [{"filename": "chart1.html", "title": "PM2.5 Trend"}]

    def test_report_with_empty_running_summary_uses_fallback(self):
        """finalize_report must not crash when running_summary is empty (uses web_research_results)."""
        from src.graph import finalize_report
        from src.state import SummaryState

        state = SummaryState(
            research_topic="test",
            running_summary="",   # empty — should fall back to web_research_results
            web_research_results=[{"content": "Delhi AQI was 278 today.", "sources": []}],
            analysis_required=False,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        config = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Fallback Report\nDerived from raw data.")

        with patch("src.graph.get_llm_client", return_value=mock_llm):
            result = finalize_report(state, config)

        assert "running_summary" in result
        assert result["running_summary"].strip() != ""
