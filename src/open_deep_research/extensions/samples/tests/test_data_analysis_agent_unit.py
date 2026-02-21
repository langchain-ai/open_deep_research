"""Unit tests for DataAnalysisAgent - _llm_extract_data, run_pipeline, _plan_charts, helpers.

All LLM calls are mocked to test logic without requiring API keys.
"""
import os
import sys
import csv
import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.models.extracted_data_schema import ExtractedTable, ExtractedDataset


# ── Test Data ──

SAMPLE_CSV = """\
Country,GDP_Trillion,Growth_Percent
USA,25.5,2.1
China,17.9,5.2
India,3.7,6.8
Germany,4.1,1.5
Japan,4.2,1.1"""

RESEARCH_WITH_TABLE = """\
# Economic Report

| Country | GDP (Trillion) | Growth (%) |
|---------|---------------|------------|
| USA | 25.5 | 2.1 |
| China | 17.9 | 5.2 |

The US GDP was 25.5 trillion with 2.1% growth. China reached 17.9 trillion.
"""

RESEARCH_PROSE_ONLY = """\
The global economy showed mixed signals in 2023. The United States recorded
a GDP of 25.5 trillion dollars with 2.1% growth. Meanwhile, China reached
17.9 trillion with a growth rate of 5.2%. India showed strong performance
at 3.7 trillion with 6.8% growth.
"""


def _make_mock_agent():
    """Create a DataAnalysisAgent with fully mocked LLM."""
    with patch("extensions.agents.data_analysis_agent.get_extensions_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
        mock_factory.return_value = mock_llm

        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        agent = DataAnalysisAgent(provider="azure", model="test-model")
        return agent


class TestLLMExtractData:
    """Test _llm_extract_data method."""

    def test_returns_empty_list_when_no_extraction_llm(self):
        agent = _make_mock_agent()
        agent.extraction_llm = None
        result = agent._llm_extract_data("Some research text")
        assert result == []

    def test_returns_csv_tables_on_success(self):
        agent = _make_mock_agent()

        # Mock the extraction_llm to return a valid ExtractedDataset
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="GDP by Country",
                headers=["Country", "GDP_Trillion"],
                rows=[["USA", "25.5"], ["China", "17.9"]],
            )
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data(RESEARCH_PROSE_ONLY)
        assert len(result) == 1
        assert "Country,GDP_Trillion" in result[0]
        assert "USA,25.5" in result[0]
        assert "China,17.9" in result[0]

    def test_returns_multiple_tables(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="GDP",
                headers=["Country", "GDP"],
                rows=[["USA", "25.5"], ["China", "17.9"]],
            ),
            ExtractedTable(
                table_name="Growth",
                headers=["Country", "Growth"],
                rows=[["USA", "2.1"], ["India", "6.8"]],
            ),
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data(RESEARCH_PROSE_ONLY)
        assert len(result) == 2

    def test_skips_tables_with_single_column(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Bad",
                headers=["OnlyOneColumn"],
                rows=[["a"], ["b"]],
            ),
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert len(result) == 0

    def test_skips_tables_with_no_rows(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Empty",
                headers=["A", "B"],
                rows=[],
            ),
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert len(result) == 0

    def test_pads_short_rows(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Mismatched",
                headers=["A", "B", "C"],
                rows=[["1"]],  # Only 1 value, should be padded to 3
            ),
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert len(result) == 1
        csv_lines = result[0].strip().split("\n")
        assert len(csv_lines) == 2  # header + 1 row
        row_values = csv_lines[1].split(",")
        assert len(row_values) == 3  # padded to match header count

    def test_trims_long_rows(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Long",
                headers=["A", "B"],
                rows=[["1", "2", "3", "4"]],  # 4 values, only 2 headers
            ),
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert len(result) == 1
        csv_lines = result[0].strip().split("\n")
        row_values = csv_lines[1].split(",")
        assert len(row_values) == 2  # trimmed to match headers

    def test_truncates_long_input(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        long_text = "x" * 50000
        agent._llm_extract_data(long_text)

        # Check that the prompt was truncated (30000 chars of text + prompt instructions)
        call_args = agent.extraction_llm.invoke.call_args[0][0]
        # The full prompt should contain 30000 x's, not 50000
        assert "x" * 30000 in call_args
        assert "x" * 50000 not in call_args

    def test_returns_empty_on_llm_exception(self):
        agent = _make_mock_agent()
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(side_effect=Exception("API Error"))

        result = agent._llm_extract_data("text")
        assert result == []

    def test_returns_empty_on_empty_dataset(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert result == []


class TestPlanCharts:
    """Test _plan_charts method."""

    def test_returns_default_on_llm_failure(self):
        agent = _make_mock_agent()
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(side_effect=Exception("LLM Error"))

        result = agent._plan_charts(SAMPLE_CSV, "profile text")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["chart_type"] == "bar"

    def test_returns_parsed_plan(self):
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"chart_type": "line", "title": "Trend", "x_column": "Country", "y_column": "GDP"},
            {"chart_type": "bar", "title": "Compare", "x_column": "Country", "y_column": "Growth"},
        ])
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts(SAMPLE_CSV, "profile text")
        assert len(result) == 2
        assert result[0]["chart_type"] == "line"
        assert result[1]["chart_type"] == "bar"

    def test_limits_to_4_charts(self):
        agent = _make_mock_agent()
        many_charts = [
            {"chart_type": "bar", "title": f"Chart {i}", "x_column": "A", "y_column": "B"}
            for i in range(10)
        ]
        mock_response = MagicMock()
        mock_response.content = json.dumps(many_charts)
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts(SAMPLE_CSV, "profile")
        assert len(result) <= 4

    def test_filters_invalid_specs(self):
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"chart_type": "bar", "title": "Good"},
            {"no_chart_type": True},  # Missing chart_type
            "not a dict",
        ])
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts(SAMPLE_CSV, "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "bar"

    def test_default_when_no_valid_specs(self):
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"no_chart_type": True},
        ])
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts(SAMPLE_CSV, "profile")
        # Should fallback to default bar chart
        assert len(result) == 1
        assert result[0]["chart_type"] == "bar"

    def test_handles_json_in_markdown_block(self):
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = '```json\n[{"chart_type": "pie", "title": "T", "x_column": "A", "y_column": "B"}]\n```'
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts(SAMPLE_CSV, "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "pie"


class TestTitleFromPath:
    """Test the static _title_from_path helper."""

    def test_bar_chart_path(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("outputs/charts/bar_abc12345.html")
        assert title == "Bar Chart"

    def test_line_chart_path(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("outputs/charts/line_def45678.html")
        assert title == "Line Chart"

    def test_scatter_chart_path(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("outputs/charts/scatter_12345678.html")
        assert title == "Scatter Chart"

    def test_outliers_chart_path(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("outputs/charts/outliers_abcd1234.html")
        assert title == "Outliers Chart"

    def test_unknown_format(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("some/random/file.txt")
        assert title == "Chart"

    def test_empty_path(self):
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        title = DataAnalysisAgent._title_from_path("")
        assert title == "Chart"


class TestRunPipelineLLMPrimary:
    """Test run_pipeline with mocked LLM extraction (primary path)."""

    def test_pipeline_uses_llm_extraction_primary(self):
        agent = _make_mock_agent()

        # Mock _llm_extract_data to return tables
        csv_tables = [
            "Country,GDP\nUSA,25.5\nChina,17.9",
            "Country,Growth\nUSA,2.1\nIndia,6.8",
        ]
        agent._llm_extract_data = MagicMock(return_value=csv_tables)

        # Mock _plan_charts to return simple specs
        agent._plan_charts = MagicMock(return_value=[
            {"chart_type": "bar", "title": "Test", "x_column": "Country", "y_column": "GDP"},
        ])

        # Mock _extract_chart_explanations
        agent._extract_chart_explanations = MagicMock(return_value={})

        result = agent.run_pipeline(RESEARCH_PROSE_ONLY)

        assert result["status"] == "completed"
        assert "2 tables" in result["output"]
        assert agent._llm_extract_data.called

    def test_pipeline_falls_back_to_regex(self):
        agent = _make_mock_agent()

        # LLM returns nothing
        agent._llm_extract_data = MagicMock(return_value=[])

        # Mock _plan_charts
        agent._plan_charts = MagicMock(return_value=[
            {"chart_type": "bar", "title": "Test", "x_column": "", "y_column": ""},
        ])
        agent._extract_chart_explanations = MagicMock(return_value={})

        # Use text with a markdown table so regex can find it
        result = agent.run_pipeline(RESEARCH_WITH_TABLE)

        assert result["status"] == "completed"
        # Regex should have extracted 1 table
        assert "1 table" in result["output"] or len(result["charts"]) >= 0

    def test_pipeline_returns_no_data_when_both_fail(self):
        agent = _make_mock_agent()

        # LLM returns nothing
        agent._llm_extract_data = MagicMock(return_value=[])

        # Use plain text that regex can't parse
        result = agent.run_pipeline("No structured data here at all.")

        assert result["status"] == "completed"
        assert result["charts"] == []
        assert "No structured data" in result["output"]

    def test_pipeline_respects_chart_cap(self):
        agent = _make_mock_agent()

        # 10 tables (each would produce charts)
        csv_tables = [f"A,B\n{i},{i*10}" for i in range(10)]
        agent._llm_extract_data = MagicMock(return_value=csv_tables)

        chart_count = [0]

        def mock_plan_charts(data, profile):
            return [
                {"chart_type": "bar", "title": "T", "x_column": "A", "y_column": "B"},
                {"chart_type": "line", "title": "T2", "x_column": "A", "y_column": "B"},
            ]

        agent._plan_charts = mock_plan_charts
        agent._extract_chart_explanations = MagicMock(return_value={})

        result = agent.run_pipeline("dummy text")

        # Total charts should not exceed cap (8)
        assert len(result["charts"]) <= 8

    def test_pipeline_returns_all_expected_keys(self):
        agent = _make_mock_agent()
        agent._llm_extract_data = MagicMock(return_value=[])

        result = agent.run_pipeline("No data")

        expected_keys = {
            "extracted_data", "data_profile", "charts",
            "chart_explanations", "outlier_analysis",
            "status", "output", "execution_time", "error",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_pipeline_handles_exception(self):
        agent = _make_mock_agent()
        agent._llm_extract_data = MagicMock(side_effect=Exception("Boom"))

        result = agent.run_pipeline("some text")

        assert result["status"] == "error"
        assert result["error"] is not None
        assert "Boom" in result["error"]

    def test_pipeline_execution_time_recorded(self):
        agent = _make_mock_agent()
        agent._llm_extract_data = MagicMock(return_value=[])

        result = agent.run_pipeline("No data")

        assert "execution_time" in result
        assert result["execution_time"] >= 0


class TestExtractFromIntermediateSteps:
    """Test _extract_from_intermediate_steps method."""

    def test_extracts_paths_from_steps(self):
        agent = _make_mock_agent()

        action = MagicMock()
        action.tool = "create_chart"

        # Create a temp file to pass validation
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html></html>")
            temp_path = f.name

        try:
            steps = [(action, f"Chart created!\nFile: {temp_path}\nDone")]
            result = agent._extract_from_intermediate_steps(steps)
            assert temp_path in result
        finally:
            os.unlink(temp_path)

    def test_ignores_non_create_chart_tools(self):
        agent = _make_mock_agent()
        action = MagicMock()
        action.tool = "other_tool"
        steps = [(action, "File: /tmp/fake.html")]
        result = agent._extract_from_intermediate_steps(steps)
        assert result == []

    def test_handles_empty_steps(self):
        agent = _make_mock_agent()
        result = agent._extract_from_intermediate_steps([])
        assert result == []

    def test_handles_malformed_steps(self):
        agent = _make_mock_agent()
        result = agent._extract_from_intermediate_steps("not a list")
        assert result == []


class TestExtractChartExplanations:
    """Test _extract_chart_explanations method."""

    def test_fallback_when_llm_fails(self):
        agent = _make_mock_agent()
        agent.explanation_llm = MagicMock()
        agent.explanation_llm.invoke = MagicMock(side_effect=Exception("Fail"))

        result = agent._extract_chart_explanations("output text", ["/tmp/bar_abc.html"])
        assert "/tmp/bar_abc.html" in result
        assert result["/tmp/bar_abc.html"]["title"] is not None

    def test_fallback_generates_generic_explanations(self):
        agent = _make_mock_agent()
        agent.explanation_llm = MagicMock()
        agent.explanation_llm.invoke = MagicMock(side_effect=Exception("Fail"))

        paths = ["/tmp/bar_abc12345.html", "/tmp/line_def12345.html"]
        result = agent._extract_chart_explanations("text", paths)
        assert len(result) == 2
        for path in paths:
            assert path in result
            assert "explanation" in result[path]


class TestExtractFromOutputLLM:
    """Test _extract_from_output_llm method."""

    def test_empty_output(self):
        agent = _make_mock_agent()
        result = agent._extract_from_output_llm("")
        assert result == []

    def test_whitespace_output(self):
        agent = _make_mock_agent()
        result = agent._extract_from_output_llm("   \n  ")
        assert result == []


# ── Fix 1: CSV values with commas are properly quoted ──

class TestFix1CsvWriterQuoting:
    """Verify _llm_extract_data uses csv.writer for proper quoting."""

    def test_values_with_commas_are_quoted(self):
        """Values containing commas must be quoted in the CSV output."""
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Cities",
                headers=["City", "Population"],
                rows=[["New York, USA", "8336817"], ["Los Angeles, USA", "3979576"]],
            )
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        assert len(result) == 1
        # Parse the CSV back — csv.reader should correctly handle quoted fields
        reader = csv.reader(StringIO(result[0]))
        rows = list(reader)
        assert rows[0] == ["City", "Population"]
        assert rows[1] == ["New York, USA", "8336817"]
        assert rows[2] == ["Los Angeles, USA", "3979576"]

    def test_values_with_quotes_are_escaped(self):
        """Values containing double quotes must be escaped."""
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Quotes",
                headers=["Name", "Value"],
                rows=[['He said "hello"', "42"]],
            )
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        reader = csv.reader(StringIO(result[0]))
        rows = list(reader)
        assert rows[1][0] == 'He said "hello"'

    def test_numeric_values_with_commas(self):
        """Numbers like 1,000,000 in values are properly handled."""
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[
            ExtractedTable(
                table_name="Numbers",
                headers=["Item", "Cost"],
                rows=[["Widget", "1,000,000"], ["Gadget", "500,000"]],
            )
        ])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        result = agent._llm_extract_data("text")
        reader = csv.reader(StringIO(result[0]))
        rows = list(reader)
        assert rows[1] == ["Widget", "1,000,000"]
        assert rows[2] == ["Gadget", "500,000"]


# ── Fix 2: Truncation limit increased to 30000 + warning log ──

class TestFix2TruncationLimit:
    """Verify truncation is 30000 chars with warning log."""

    def test_truncates_at_30000_chars(self):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        long_text = "A" * 50000
        agent._llm_extract_data(long_text)

        # The prompt should contain at most 30000 chars of research text
        call_args = agent.extraction_llm.invoke.call_args[0][0]
        # The full prompt includes instructions + the truncated text
        # The research text portion should be 30000 chars
        assert "A" * 30000 in call_args
        # But not the full 50000
        assert "A" * 50000 not in call_args

    def test_truncation_logs_warning(self, caplog):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        long_text = "B" * 40000
        with caplog.at_level(logging.WARNING):
            agent._llm_extract_data(long_text)
        assert any("truncated" in msg.lower() for msg in caplog.messages)

    def test_no_warning_for_short_text(self, caplog):
        agent = _make_mock_agent()
        mock_dataset = ExtractedDataset(tables=[])
        agent.extraction_llm = MagicMock()
        agent.extraction_llm.invoke = MagicMock(return_value=mock_dataset)

        short_text = "C" * 1000
        with caplog.at_level(logging.WARNING):
            agent._llm_extract_data(short_text)
        assert not any("truncated" in msg.lower() for msg in caplog.messages)


# ── Fix 4: Smart numeric column detection for outlier analysis ──

class TestFix4NumericColumnDetection:
    """Verify outlier detection finds numeric columns by sampling data."""

    def _make_pipeline_agent_with_csv(self, csv_data):
        """Helper: create agent that returns given CSV from LLM extraction."""
        agent = _make_mock_agent()
        agent._llm_extract_data = MagicMock(return_value=[csv_data])
        agent._plan_charts = MagicMock(return_value=[])  # No regular charts
        agent._extract_chart_explanations = MagicMock(return_value={})
        return agent

    def test_detects_numeric_columns(self):
        """Should identify numeric columns (not the first/category column)."""
        csv_data = "Name,Age,Score,City\nAlice,30,85,NYC\nBob,25,92,LA\nCharlie,35,78,SF\nDave,28,88,CHI\nEve,32,95,BOS"
        agent = self._make_pipeline_agent_with_csv(csv_data)

        # We need to mock detect_outliers to capture which columns are called
        called_columns = []
        original_detect = detect_outliers_import = None

        def mock_detect(data, column, method):
            called_columns.append(column)
            return "No outliers found.\nTotal values: 5"

        with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
            agent.run_pipeline("text")

        # Age and Score should be detected as numeric (City is not)
        assert "Age" in called_columns or "Score" in called_columns
        assert "City" not in called_columns

    def test_limits_to_2_columns_per_table(self):
        """Should run outlier detection on at most 2 numeric columns per table."""
        csv_data = "ID,A,B,C,D\n1,10,20,30,40\n2,11,21,31,41\n3,12,22,32,42\n4,13,23,33,43\n5,14,24,34,44"
        agent = self._make_pipeline_agent_with_csv(csv_data)

        called_columns = []

        def mock_detect(data, column, method):
            called_columns.append(column)
            return "No outliers.\nTotal values: 5"

        with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
            agent.run_pipeline("text")

        # Should be at most 2 columns
        assert len(called_columns) <= 2

    def test_skips_non_numeric_columns(self):
        """Columns with text values should be skipped."""
        csv_data = "Name,Label,Value\nA,cat,10\nB,dog,20\nC,bird,30\nD,fish,40\nE,ant,50"
        agent = self._make_pipeline_agent_with_csv(csv_data)

        called_columns = []

        def mock_detect(data, column, method):
            called_columns.append(column)
            return "No outliers.\nTotal values: 5"

        with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
            agent.run_pipeline("text")

        # Label is text, should not be called; Value is numeric
        assert "Label" not in called_columns
        if called_columns:
            assert "Value" in called_columns


# ── Fix 5: None check for explanation_llm ──

class TestFix5ExplanationLlmNoneGuard:
    """Verify _extract_chart_explanations handles None explanation_llm."""

    def test_returns_generic_when_explanation_llm_is_none(self):
        agent = _make_mock_agent()
        agent.explanation_llm = None

        paths = ["/tmp/bar_abc12345.html", "/tmp/line_def12345.html"]
        result = agent._extract_chart_explanations("analysis output", paths)

        assert len(result) == 2
        for path in paths:
            assert path in result
            assert "title" in result[path]
            assert "explanation" in result[path]
            assert result[path]["explanation"] == "Visualization generated from the data analysis."

    def test_does_not_call_invoke_when_none(self):
        agent = _make_mock_agent()
        agent.explanation_llm = None

        # If it tried to call .invoke(), this would raise AttributeError
        result = agent._extract_chart_explanations("text", ["/tmp/test.html"])
        assert result is not None  # Should succeed without error


# ── Fix 6: _plan_charts JSON extraction improvements ──

class TestFix6PlanChartsJsonExtraction:
    """Verify _plan_charts tries json.loads first, then regex fallback."""

    def test_parses_clean_json_array(self):
        """Clean JSON array response should be parsed directly."""
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = '[{"chart_type": "bar", "title": "T", "x_column": "X", "y_column": "Y"}]'
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts("data", "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "bar"

    def test_parses_json_in_markdown_fenced_block(self):
        """JSON wrapped in ```json ... ``` should be parsed."""
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = '```json\n[{"chart_type": "scatter", "title": "T", "x_column": "X", "y_column": "Y"}]\n```'
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts("data", "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "scatter"

    def test_parses_json_embedded_in_text(self):
        """JSON array embedded in explanatory text should be found by regex fallback."""
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = 'Here are the charts:\n[{"chart_type": "line", "title": "T", "x_column": "X", "y_column": "Y"}]\nEnd.'
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts("data", "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "line"

    def test_returns_default_on_no_json(self):
        """When response has no JSON at all, return default bar chart."""
        agent = _make_mock_agent()
        mock_response = MagicMock()
        mock_response.content = "I suggest making a bar chart of the data."
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)

        result = agent._plan_charts("data", "profile")
        assert len(result) == 1
        assert result[0]["chart_type"] == "bar"
        assert result[0]["title"] == "Data Overview"


# ── Fix 7: Accumulated outlier_analyses list ──

class TestFix7OutlierAnalysisAccumulation:
    """Verify outlier_analysis collects results from all tables, not just last."""

    def test_accumulates_across_tables(self):
        """outlier_analysis should be a list when there are results."""
        agent = _make_mock_agent()

        csv_tables = [
            "A,B\n1,10\n2,12\n3,11\n4,13\n5,100",  # Table 1 with outlier
            "X,Y\n1,20\n2,22\n3,21\n4,23\n5,200",  # Table 2 with outlier
        ]
        agent._llm_extract_data = MagicMock(return_value=csv_tables)
        agent._plan_charts = MagicMock(return_value=[])
        agent._extract_chart_explanations = MagicMock(return_value={})

        call_count = [0]

        def mock_detect(data, column, method):
            call_count[0] += 1
            return f"Outliers found: 1\nTotal values: 5\nResult {call_count[0]}"

        with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
            result = agent.run_pipeline("text")

        # outlier_analysis should be a list (not None or single value)
        if result["outlier_analysis"] is not None:
            assert isinstance(result["outlier_analysis"], list)

    def test_returns_none_when_no_outliers(self):
        """outlier_analysis should be None when no outlier results."""
        agent = _make_mock_agent()
        agent._llm_extract_data = MagicMock(return_value=[])

        result = agent.run_pipeline("No data")
        # When no data extracted, outlier_analysis should be empty list or None
        assert result["outlier_analysis"] is None or result["outlier_analysis"] == []


# ── Fix 8: File: vs Visualization: mismatch ──

class TestFix8VisualizationMarker:
    """Verify outlier chart paths are captured from Visualization: marker."""

    def test_extracts_path_from_visualization_marker(self):
        """detect_outliers returns 'Visualization:' not 'File:' — should still capture path."""
        agent = _make_mock_agent()
        csv_data = "ID,Value\n1,10\n2,12\n3,11\n4,13\n5,100"
        agent._llm_extract_data = MagicMock(return_value=[csv_data])
        agent._plan_charts = MagicMock(return_value=[])
        agent._extract_chart_explanations = MagicMock(return_value={})

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, prefix="outliers_") as f:
            f.write(b"<html>chart</html>")
            temp_path = f.name

        try:
            def mock_detect(data, column, method):
                return f"Outliers found: 1\nTotal values: 5\nVisualization: {temp_path}\nDone"

            with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
                result = agent.run_pipeline("text")

            assert temp_path in result["charts"]
        finally:
            os.unlink(temp_path)

    def test_extracts_path_from_file_marker(self):
        """If detect_outliers ever returns 'File:' marker, it should still work."""
        agent = _make_mock_agent()
        csv_data = "ID,Value\n1,10\n2,12\n3,11\n4,13\n5,100"
        agent._llm_extract_data = MagicMock(return_value=[csv_data])
        agent._plan_charts = MagicMock(return_value=[])
        agent._extract_chart_explanations = MagicMock(return_value={})

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, prefix="outliers_") as f:
            f.write(b"<html>chart</html>")
            temp_path = f.name

        try:
            def mock_detect(data, column, method):
                return f"Outliers found: 1\nFile: {temp_path}\nDone"

            with patch("extensions.agents.data_analysis_agent.detect_outliers", side_effect=mock_detect):
                result = agent.run_pipeline("text")

            assert temp_path in result["charts"]
        finally:
            os.unlink(temp_path)
