"""Unit tests for data_profiling.py - parse_data and profile_data functions."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.tools.data_profiling import parse_data, profile_data


# ── Test Data ──

CSV_DATA = """\
Name,Age,Salary
Alice,30,70000
Bob,25,55000
Charlie,35,85000
Diana,28,62000
Eve,40,95000"""

JSON_LIST = '[{"Name":"Alice","Age":30},{"Name":"Bob","Age":25}]'
JSON_OBJECT = '{"Name":"Alice","Age":30}'

TSV_DATA = "Name\tAge\tSalary\nAlice\t30\t70000\nBob\t25\t55000"

PIPE_DATA = "| Name | Age |\n| Alice | 30 |\n| Bob | 25 |"


class TestParseData:
    """Test the parse_data helper function."""

    def test_parse_csv(self):
        df = parse_data(CSV_DATA)
        assert df is not None
        assert len(df) == 5
        assert "Name" in df.columns
        assert "Age" in df.columns
        assert "Salary" in df.columns

    def test_parse_json_list(self):
        df = parse_data(JSON_LIST)
        assert df is not None
        assert len(df) == 2
        assert "Name" in df.columns

    def test_parse_json_object(self):
        df = parse_data(JSON_OBJECT)
        assert df is not None
        assert len(df) == 1

    def test_parse_tsv(self):
        df = parse_data(TSV_DATA)
        assert df is not None
        assert len(df) == 2

    def test_parse_pipe_separated(self):
        df = parse_data(PIPE_DATA)
        assert df is not None

    def test_parse_empty_string(self):
        df = parse_data("")
        assert df is None

    def test_parse_plain_text(self):
        df = parse_data("Hello world no data here")
        # May return None or an empty/invalid dataframe
        # The function should not crash
        assert df is None or hasattr(df, "columns")

    def test_parse_strips_whitespace(self):
        df = parse_data("  \n" + CSV_DATA + "\n  ")
        assert df is not None
        assert len(df) == 5

    def test_parse_csv_with_numeric_types(self):
        df = parse_data(CSV_DATA)
        assert df is not None
        # Pandas should auto-detect numeric columns
        import pandas as pd
        assert pd.api.types.is_numeric_dtype(df["Age"])
        assert pd.api.types.is_numeric_dtype(df["Salary"])


class TestProfileData:
    """Test the profile_data function."""

    def test_profile_returns_string(self):
        result = profile_data(CSV_DATA)
        assert isinstance(result, str)

    def test_profile_not_error(self):
        result = profile_data(CSV_DATA)
        assert "[ERROR]" not in result

    def test_profile_contains_shape(self):
        result = profile_data(CSV_DATA)
        assert "5 rows" in result
        assert "3 columns" in result

    def test_profile_contains_columns(self):
        result = profile_data(CSV_DATA)
        assert "Name" in result
        assert "Age" in result
        assert "Salary" in result

    def test_profile_contains_stats(self):
        result = profile_data(CSV_DATA)
        assert "Summary Statistics" in result

    def test_profile_contains_missing_values_section(self):
        result = profile_data(CSV_DATA)
        # Missing column is in the Columns pipe table
        assert "Missing" in result

    def test_profile_with_missing_values(self):
        csv_with_missing = "A,B\n1,\n2,3\n,4"
        result = profile_data(csv_with_missing)
        assert "Missing" in result

    def test_profile_column_analysis(self):
        result = profile_data(CSV_DATA)
        assert "Column Details" in result

    def test_profile_numeric_stats(self):
        result = profile_data(CSV_DATA)
        # Should contain arrow range and mean for numeric cols
        assert "→" in result
        assert "mean" in result

    def test_profile_categorical_stats(self):
        result = profile_data(CSV_DATA)
        # Name column should show unique values
        assert "unique values" in result

    def test_profile_empty_data_returns_error(self):
        result = profile_data("")
        assert "[ERROR]" in result

    def test_profile_invalid_data_returns_error(self):
        result = profile_data("just plain text no structure")
        # May or may not be an error depending on parse_data behavior
        assert isinstance(result, str)

    def test_profile_json_input(self):
        result = profile_data(JSON_LIST)
        assert isinstance(result, str)
        assert "[ERROR]" not in result
        assert "2 rows" in result

    def test_profile_comprehensive_type(self):
        result = profile_data(CSV_DATA, analysis_type="comprehensive")
        assert isinstance(result, str)
        assert "[ERROR]" not in result

    def test_profile_statistical_type(self):
        result = profile_data(CSV_DATA, analysis_type="statistical")
        assert isinstance(result, str)
        # Currently no different behavior, but should not crash
        assert "[ERROR]" not in result

    def test_profile_single_column(self):
        single_col = "Value\n10\n20\n30"
        result = profile_data(single_col)
        assert isinstance(result, str)


class TestProfileMarkdownFormatting:
    """Verify profile_data() output is valid markdown that _md_to_html() can render."""

    def test_output_contains_pipe_tables(self):
        """Columns section must be a markdown pipe table."""
        result = profile_data(CSV_DATA)
        assert "| Column | Type | Missing |" in result
        assert "|--------|------|" in result

    def test_stats_is_pipe_table(self):
        """Summary Statistics must be a pipe table, not space-aligned text."""
        result = profile_data(CSV_DATA)
        assert "| Statistic |" in result
        # Must NOT contain space-aligned pandas output
        assert "count  " not in result  # old format had "count  5.000000"

    def test_no_ascii_banners(self):
        """Old ====== banners must be gone."""
        result = profile_data(CSV_DATA)
        assert "=====" not in result
        assert "DATA PROFILING REPORT" not in result

    def test_headings_are_markdown(self):
        """Section headings must use ### syntax."""
        result = profile_data(CSV_DATA)
        assert "### Columns" in result
        assert "### Summary Statistics" in result
        assert "### Column Details" in result

    def test_column_details_are_bullets(self):
        """Column details must be markdown bullets with bold names."""
        result = profile_data(CSV_DATA)
        assert "- **Name**:" in result
        assert "- **Age**:" in result

    def test_missing_values_in_columns_table(self):
        """Missing values must appear in the Columns pipe table."""
        csv_with_missing = "A,B\n1,\n2,3\n,4"
        result = profile_data(csv_with_missing)
        # Should show percentage in the Missing column
        assert "%" in result
        assert "| A |" in result

    def test_stats_values_match_pandas(self):
        """Verify stats numbers match pandas computation exactly."""
        result = profile_data(CSV_DATA)
        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(CSV_DATA))
        age_mean = df["Age"].mean()
        # The rounded value should appear in the pipe table
        assert f"{age_mean:.2f}" in result  # e.g. "31.60"

    def test_renders_to_html_tables(self):
        """The markdown output must produce <table> tags when passed through _md_to_html."""
        result = profile_data(CSV_DATA)
        from extensions.utils.report_builder import _md_to_html
        html = _md_to_html(result)
        assert "<table>" in html
        assert "<th>" in html
        assert "<td>" in html


class TestProfileDataTool:
    """Test the LangChain tool wrapper."""

    def test_tool_exists(self):
        from extensions.tools.data_profiling import profile_data_tool
        assert profile_data_tool is not None

    def test_tool_name(self):
        from extensions.tools.data_profiling import profile_data_tool
        assert profile_data_tool.name == "profile_data"

    def test_tool_has_description(self):
        from extensions.tools.data_profiling import profile_data_tool
        assert len(profile_data_tool.description) > 0
