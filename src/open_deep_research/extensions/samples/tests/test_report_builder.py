"""Tests for the HTML report builder."""
import os
import sys
import pytest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestBuildHtmlReport:
    """Test build_html_report function."""

    def test_report_creates_html_file(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="# Research Findings\n\nSome research content.",
            analysis_output="Analysis complete.",
            figures=[],
            chart_explanations={},
            sources=["https://example.com"],
            query="test query",
            conversation_id="abcd1234",
            src_dir=tmp_path,
        )
        assert Path(report_path).exists()
        assert report_path.endswith(".html")

    def test_report_contains_query(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="What is renewable energy?",
            conversation_id="test0001",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "What is renewable energy?" in content

    def test_report_contains_sources(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=["https://source1.com", "https://source2.com"],
            query="test",
            conversation_id="test0002",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "https://source1.com" in content
        assert "https://source2.com" in content

    def test_report_contains_sub_queries(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            sub_queries=["Subtopic A", "Subtopic B", "Subtopic C"],
            conversation_id="test0003",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Subtopic A" in content
        assert "Subtopic B" in content
        assert "Sub-Queries Explored" in content

    def test_report_contains_research_findings(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="# Solar Energy Growth\n\nSolar is growing rapidly.",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="solar energy",
            conversation_id="test0004",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Research Findings" in content
        assert "Solar Energy Growth" in content

    def test_report_naming_convention(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="abcdef12-3456-7890",
            src_dir=tmp_path,
        )
        filename = Path(report_path).name
        assert filename.startswith("report_")
        assert filename.endswith(".html")
        # Should use first 8 chars of conversation_id
        assert "abcdef12" in filename

    def test_report_handles_missing_optional_fields(self, tmp_path):
        """Report should be generated even with minimal inputs."""
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Minimal report",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            src_dir=tmp_path,
        )
        assert Path(report_path).exists()

    def test_report_handles_empty_display_text(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            src_dir=tmp_path,
        )
        assert Path(report_path).exists()


class TestAnalysisResultsSection:
    """Test the new Analysis Results section with extracted data and profile."""

    def test_report_contains_extracted_data_table(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="test0010",
            src_dir=tmp_path,
            extracted_data_summary="Year,Value\n2020,10\n2021,20\n2022,30",
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Extracted Data Summary" in content
        assert "Year" in content
        assert "2020" in content
        assert "2022" in content

    def test_report_contains_data_profile(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="test0011",
            src_dir=tmp_path,
            data_profile_summary="**Shape:** 3 rows, 2 columns\n\n**Numeric columns:** Value (mean=20.0)",
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Data Profile Highlights" in content

    def test_report_analysis_results_section_header(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="test0012",
            src_dir=tmp_path,
            extracted_data_summary="Col1,Col2\nA,1\nB,2",
            data_profile_summary="2 rows, 2 columns",
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Analysis Results" in content

    def test_report_shows_first_10_rows_only(self, tmp_path):
        """Large datasets should show only first 10 rows in the preview."""
        from extensions.utils.report_builder import build_html_report

        # Generate 20 rows of data
        rows = ["Year,Value"] + [f"{2000+i},{i*10}" for i in range(20)]
        csv_data = "\n".join(rows)

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="test0013",
            src_dir=tmp_path,
            extracted_data_summary=csv_data,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "Showing first 10 of 20 rows" in content

    def test_report_no_analysis_results_without_data(self, tmp_path):
        """Without extracted_data or data_profile, Analysis Results section should not appear."""
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="Research content",
            analysis_output="Some analysis text",
            figures=[],
            chart_explanations={},
            sources=[],
            query="test",
            conversation_id="test0014",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        # Should fall back to legacy "Data Analysis" section
        assert "Data Analysis" in content


class TestReportHTMLValidity:
    """Basic HTML structure checks."""

    def test_report_has_doctype(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")

    def test_report_has_closing_tags(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "</html>" in content
        assert "</body>" in content

    def test_report_includes_plotly_js(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        assert "plotly-latest.min.js" in content

    def test_report_escapes_html_in_query(self, tmp_path):
        from extensions.utils.report_builder import build_html_report

        report_path = build_html_report(
            display_text="content",
            analysis_output="",
            figures=[],
            chart_explanations={},
            sources=[],
            query='<script>alert("xss")</script>',
            conversation_id="test0020",
            src_dir=tmp_path,
        )
        content = Path(report_path).read_text(encoding="utf-8")
        # HTML-escaped, not raw script tag
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content
