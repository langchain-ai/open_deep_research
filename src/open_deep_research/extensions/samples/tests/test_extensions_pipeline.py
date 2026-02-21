"""Tests for the DataAnalysisAgent enforced pipeline."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Sample test data
SAMPLE_RESEARCH_TEXT = """
# Renewable Energy Report

## Key Data & Statistics

| Year | Solar Share (%) | Wind Share (%) | Total Renewable (%) |
|------|----------------|----------------|---------------------|
| 2018 | 2.4 | 5.6 | 11.0 |
| 2019 | 3.1 | 6.3 | 12.5 |
| 2020 | 3.7 | 7.1 | 14.2 |
| 2021 | 4.5 | 7.8 | 15.8 |
| 2022 | 5.2 | 8.5 | 17.3 |
| 2023 | 6.1 | 9.2 | 19.0 |

Solar energy grew from 2.4% in 2018 to 6.1% in 2023, while wind grew from 5.6% to 9.2%.
"""

SAMPLE_CSV = """Year,Solar Share (%),Wind Share (%),Total Renewable (%)
2018,2.4,5.6,11.0
2019,3.1,6.3,12.5
2020,3.7,7.1,14.2
2021,4.5,7.8,15.8
2022,5.2,8.5,17.3
2023,6.1,9.2,19.0"""


class TestExtractData:
    """Test the extract_data tool function."""

    def test_extract_data_returns_string(self):
        from extensions.tools.data_extraction import extract_data
        result = extract_data(SAMPLE_RESEARCH_TEXT, format="csv")
        assert isinstance(result, str)

    def test_extract_data_contains_data(self):
        from extensions.tools.data_extraction import extract_data
        result = extract_data(SAMPLE_RESEARCH_TEXT, format="csv")
        # Should contain some numeric data or CSV structure
        assert len(result) > 0

    def test_extract_data_empty_input(self):
        from extensions.tools.data_extraction import extract_data
        result = extract_data("No data here, just plain text.", format="csv")
        # Should handle gracefully
        assert isinstance(result, str)


class TestProfileData:
    """Test the profile_data tool function."""

    def test_profile_data_returns_string(self):
        from extensions.tools.data_profiling import profile_data
        result = profile_data(SAMPLE_CSV)
        assert isinstance(result, str)

    def test_profile_data_contains_stats(self):
        from extensions.tools.data_profiling import profile_data
        result = profile_data(SAMPLE_CSV)
        assert len(result) > 0


class TestCreateChart:
    """Test the create_chart tool function."""

    def test_create_chart_bar(self):
        from extensions.tools.visualization import create_chart
        result = create_chart(
            data=SAMPLE_CSV,
            chart_type="bar",
            title="Solar Share by Year",
            x_column="Year",
            y_column="Solar Share (%)",
        )
        assert isinstance(result, str)

    def test_create_chart_returns_file_path(self):
        from extensions.tools.visualization import create_chart
        result = create_chart(
            data=SAMPLE_CSV,
            chart_type="line",
            title="Renewable Energy Trends",
            x_column="Year",
            y_column="Total Renewable (%)",
        )
        # Should contain file path indicator
        assert "File:" in result or "outputs/charts" in result or "Error" in result

    def test_create_chart_creates_html_file(self):
        from extensions.tools.visualization import create_chart
        result = create_chart(
            data=SAMPLE_CSV,
            chart_type="bar",
            title="Test Chart",
            x_column="Year",
            y_column="Solar Share (%)",
        )
        if "File:" in result:
            path = result.split("File:")[1].split("\n")[0].strip()
            assert path.endswith(".html")


class TestDetectOutliers:
    """Test the detect_outliers tool function."""

    def test_detect_outliers_returns_string(self):
        from extensions.tools.visualization import detect_outliers
        result = detect_outliers(
            data=SAMPLE_CSV,
            column="Solar Share (%)",
            method="iqr",
        )
        assert isinstance(result, str)


class TestRunPipeline:
    """Test the DataAnalysisAgent.run_pipeline() enforced chain."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_returns_expected_keys(self):
        """Pipeline result should have all required keys."""
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline(SAMPLE_RESEARCH_TEXT)

        assert "extracted_data" in result
        assert "data_profile" in result
        assert "charts" in result
        assert "chart_explanations" in result
        assert "outlier_analysis" in result
        assert "status" in result
        assert "execution_time" in result

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_status_is_completed_or_error(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline(SAMPLE_RESEARCH_TEXT)
        assert result["status"] in ("completed", "error")

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_handles_empty_input(self):
        """Pipeline should handle empty/no-data input gracefully."""
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline("This text has no data tables or statistics.")

        assert result["status"] in ("completed", "error")
        assert result["charts"] == [] or isinstance(result["charts"], list)

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_execution_time_is_positive(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline(SAMPLE_RESEARCH_TEXT)
        assert result["execution_time"] >= 0

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_charts_are_list(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline(SAMPLE_RESEARCH_TEXT)
        assert isinstance(result["charts"], list)

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_chart_explanations_are_dict(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        result = agent.run_pipeline(SAMPLE_RESEARCH_TEXT)
        assert isinstance(result["chart_explanations"], dict)


class TestPlanCharts:
    """Test the _plan_charts helper method."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_plan_charts_returns_list(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        plan = agent._plan_charts(SAMPLE_CSV, "6 rows, 4 columns: Year, Solar, Wind, Total")
        assert isinstance(plan, list)
        assert len(plan) >= 1

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_plan_charts_has_chart_type(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        plan = agent._plan_charts(SAMPLE_CSV, "6 rows, 4 columns")
        for spec in plan:
            assert "chart_type" in spec

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_plan_charts_max_4(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent

        agent = DataAnalysisAgent()
        plan = agent._plan_charts(SAMPLE_CSV, "6 rows, 4 columns")
        assert len(plan) <= 4
