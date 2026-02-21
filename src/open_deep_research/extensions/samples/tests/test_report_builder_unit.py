"""Unit tests for report_builder.py - build_html_report function."""
import os
import sys
import html as html_lib
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMdToHtml:
    """Test the internal _md_to_html helper."""

    def test_empty_input(self):
        from extensions.utils.report_builder import _md_to_html
        assert _md_to_html("") == ""

    def test_none_input(self):
        from extensions.utils.report_builder import _md_to_html
        assert _md_to_html(None) == ""

    def test_plain_text(self):
        from extensions.utils.report_builder import _md_to_html
        result = _md_to_html("Hello world")
        assert "<p>" in result
        assert "Hello world" in result

    def test_heading(self):
        from extensions.utils.report_builder import _md_to_html
        result = _md_to_html("# Title")
        assert "<h1" in result

    def test_bold(self):
        from extensions.utils.report_builder import _md_to_html
        result = _md_to_html("**bold text**")
        assert "<strong>" in result

    def test_code_block(self):
        from extensions.utils.report_builder import _md_to_html
        result = _md_to_html("```\ncode here\n```")
        assert "<code>" in result

    def test_table(self):
        from extensions.utils.report_builder import _md_to_html
        md_table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = _md_to_html(md_table)
        assert "<table" in result


class TestBuildHtmlReport:
    """Test the build_html_report function."""

    def _build_report(self, **kwargs):
        from extensions.utils.report_builder import build_html_report
        defaults = {
            "display_text": "# Test Research\nSome findings.",
            "analysis_output": "",
            "figures": [],
            "chart_explanations": {},
            "sources": [],
        }
        defaults.update(kwargs)
        return build_html_report(**defaults)

    def test_returns_file_path(self):
        result = self._build_report()
        assert isinstance(result, str)
        assert result.endswith(".html")

    def test_creates_file(self):
        result = self._build_report()
        assert Path(result).exists()

    def test_file_contains_html(self):
        result = self._build_report()
        content = Path(result).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_contains_research_findings(self):
        result = self._build_report(display_text="Key finding: important data point")
        content = Path(result).read_text(encoding="utf-8")
        assert "Research Findings" in content

    def test_contains_query(self):
        result = self._build_report(query="What is GDP growth?")
        content = Path(result).read_text(encoding="utf-8")
        assert "What is GDP growth?" in content

    def test_escapes_query_xss(self):
        result = self._build_report(query='<script>alert("xss")</script>')
        content = Path(result).read_text(encoding="utf-8")
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_contains_sources(self):
        result = self._build_report(sources=["https://example.com", "https://test.org"])
        content = Path(result).read_text(encoding="utf-8")
        assert "Sources" in content
        assert "https://example.com" in content
        assert "https://test.org" in content

    def test_contains_sub_queries(self):
        result = self._build_report(sub_queries=["Query 1", "Query 2"])
        content = Path(result).read_text(encoding="utf-8")
        assert "Sub-Queries" in content
        assert "Query 1" in content

    def test_conversation_id_in_filename(self):
        result = self._build_report(conversation_id="abc12345-test")
        assert "abc12345" in Path(result).name

    def test_no_analysis_section_without_data(self):
        result = self._build_report()
        content = Path(result).read_text(encoding="utf-8")
        # No analysis section when no extracted data or figures
        assert "Analysis Results" not in content

    def test_analysis_section_with_extracted_data(self):
        csv_data = "Country,GDP\nUSA,25.5\nChina,17.9"
        result = self._build_report(extracted_data_summary=csv_data)
        content = Path(result).read_text(encoding="utf-8")
        assert "Analysis Results" in content
        assert "Extracted Data Summary" in content
        assert "USA" in content

    def test_analysis_section_with_profile(self):
        result = self._build_report(
            extracted_data_summary="A,B\n1,2",
            data_profile_summary="Shape: 1 rows x 2 columns",
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "Data Profile Highlights" in content

    def test_extracted_data_table_limited_to_10_rows(self):
        rows = "\n".join(f"{i},{i*10}" for i in range(20))
        csv_data = f"ID,Value\n{rows}"
        result = self._build_report(extracted_data_summary=csv_data)
        content = Path(result).read_text(encoding="utf-8")
        assert "Showing first 10 of 20 rows" in content

    def test_plotly_js_included(self):
        result = self._build_report()
        content = Path(result).read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_report_directory_created(self):
        result = self._build_report()
        report_dir = Path(result).parent
        assert report_dir.exists()
        assert report_dir.name == "reports"


class TestBuildHtmlReportWithCharts:
    """Test report building with chart files."""

    def _build_report_with_charts(self, chart_paths, chart_explanations=None):
        from extensions.utils.report_builder import build_html_report
        return build_html_report(
            display_text="Research text",
            analysis_output="Analysis text",
            figures=chart_paths,
            chart_explanations=chart_explanations or {},
            sources=[],
            extracted_data_summary="A,B\n1,2",
        )

    @patch("extensions.utils.report_builder.load_plotly_figure")
    @patch("extensions.utils.report_builder.figure_to_html")
    def test_charts_section_rendered(self, mock_fig_html, mock_load_fig):
        mock_load_fig.return_value = MagicMock()
        mock_fig_html.return_value = "<div>Mock Chart</div>"

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html></html>")
            chart_path = f.name

        try:
            explanations = {
                chart_path: {"title": "GDP Chart", "explanation": "Shows GDP trends"}
            }
            result = self._build_report_with_charts([chart_path], explanations)
            content = Path(result).read_text(encoding="utf-8")
            assert "Visualizations" in content
            assert "GDP Chart" in content
            assert "Shows GDP trends" in content
        finally:
            os.unlink(chart_path)

    @patch("extensions.utils.report_builder.load_plotly_figure")
    def test_missing_chart_file_handled(self, mock_load_fig):
        mock_load_fig.return_value = None  # File not found

        result = self._build_report_with_charts(["/nonexistent/chart.html"])
        # Should not crash
        assert Path(result).exists()
