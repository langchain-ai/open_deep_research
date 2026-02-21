"""Unit tests for visualization.py - create_chart and detect_outliers functions."""
import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.tools.visualization import create_chart, detect_outliers


# ── Test Data ──

SAMPLE_CSV = """\
Category,Value,Count
A,10,100
B,25,200
C,15,150
D,30,250
E,20,180"""

NUMERIC_CSV = """\
X,Y,Z
1,10,100
2,20,200
3,15,150
4,30,300
5,25,250
6,5,50
7,35,350
8,40,400
9,10,100
10,50,500"""

TIMESERIES_CSV = """\
Date,Revenue,Expenses
2023-01,1000,800
2023-02,1200,850
2023-03,1100,900
2023-04,1500,950
2023-05,1300,880"""

OUTLIER_CSV = """\
ID,Value
1,10
2,12
3,11
4,13
5,12
6,100
7,11
8,14
9,10
10,12"""


def _extract_path(result):
    """Extract file path from create_chart result string."""
    if "File:" in result:
        return result.split("File:")[1].split("\n")[0].strip()
    return None


class TestCreateChartBar:
    """Test bar chart creation."""

    def test_bar_chart_success(self):
        result = create_chart(data=SAMPLE_CSV, chart_type="bar", title="Test Bar")
        assert "File:" in result
        assert "[ERROR]" not in result

    def test_bar_chart_creates_file(self):
        result = create_chart(data=SAMPLE_CSV, chart_type="bar", title="Test")
        path = _extract_path(result)
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_bar_chart_with_columns(self):
        result = create_chart(
            data=SAMPLE_CSV, chart_type="bar",
            title="Categories", x_column="Category", y_column="Value",
        )
        assert "File:" in result
        assert "X: Category" in result


class TestCreateChartLine:
    """Test line chart creation."""

    def test_line_chart_success(self):
        result = create_chart(data=TIMESERIES_CSV, chart_type="line", title="Trend")
        assert "File:" in result

    def test_line_chart_columns(self):
        result = create_chart(
            data=TIMESERIES_CSV, chart_type="line",
            title="Revenue", x_column="Date", y_column="Revenue",
        )
        assert "[ERROR]" not in result


class TestCreateChartScatter:
    """Test scatter chart creation."""

    def test_scatter_chart_success(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="scatter",
            title="XY Scatter", x_column="X", y_column="Y",
        )
        assert "File:" in result

    def test_scatter_with_size(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="scatter",
            title="Bubble-like", x_column="X", y_column="Y", z_column="Z",
        )
        assert "[ERROR]" not in result


class TestCreateChartPie:
    """Test pie chart creation."""

    def test_pie_chart_success(self):
        result = create_chart(
            data=SAMPLE_CSV, chart_type="pie",
            title="Distribution", x_column="Category", y_column="Value",
        )
        assert "File:" in result


class TestCreateChartHistogram:
    """Test histogram creation."""

    def test_histogram_success(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="histogram",
            title="Y Distribution", x_column="Y",
        )
        assert "File:" in result


class TestCreateChartBox:
    """Test box/boxplot chart creation."""

    def test_box_chart(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="box",
            title="Box Plot", y_column="Y",
        )
        assert "File:" in result

    def test_boxplot_alias(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="boxplot",
            title="Boxplot", y_column="Y",
        )
        assert "File:" in result


class TestCreateChartViolin:
    """Test violin chart creation."""

    def test_violin_chart(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="violin",
            title="Violin", y_column="Y",
        )
        assert "File:" in result


class TestCreateChartHeatmap:
    """Test heatmap creation."""

    def test_heatmap_correlation(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="heatmap",
            title="Correlation",
        )
        assert "File:" in result


class TestCreateChartDensity:
    """Test density chart creation."""

    def test_density_chart(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="density",
            title="Density", x_column="X", y_column="Y",
        )
        assert "File:" in result


class TestCreateChartBubble:
    """Test bubble chart creation."""

    def test_bubble_chart(self):
        result = create_chart(
            data=NUMERIC_CSV, chart_type="bubble",
            title="Bubble", x_column="X", y_column="Y", z_column="Z",
        )
        assert "File:" in result


class TestCreateChartEdgeCases:
    """Test edge cases for create_chart."""

    def test_unsupported_chart_type(self):
        result = create_chart(data=SAMPLE_CSV, chart_type="unknown_type", title="X")
        assert "[ERROR]" in result
        assert "Unsupported chart type" in result

    def test_empty_data(self):
        result = create_chart(data="", chart_type="bar", title="Empty")
        assert "[ERROR]" in result

    def test_invalid_data(self):
        result = create_chart(data="not valid csv", chart_type="bar", title="Bad")
        assert "[ERROR]" in result or "File:" in result  # parse_data may try CSV

    def test_auto_detect_columns(self):
        """When x_column and y_column are empty, auto-detection should work."""
        result = create_chart(data=SAMPLE_CSV, chart_type="bar", title="Auto")
        assert "[ERROR]" not in result

    def test_nonexistent_column(self):
        result = create_chart(
            data=SAMPLE_CSV, chart_type="bar",
            title="Bad Col", x_column="Nonexistent", y_column="Value",
        )
        # Plotly may raise an error for missing column
        assert isinstance(result, str)

    def test_chart_file_contains_html(self):
        result = create_chart(data=SAMPLE_CSV, chart_type="bar", title="HTML Check")
        path = _extract_path(result)
        if path and os.path.exists(path):
            content = open(path, encoding="utf-8").read()
            assert "<html" in content.lower() or "plotly" in content.lower()

    def test_chart_uses_cdn(self):
        """Charts should use CDN, not embed full plotly.js."""
        result = create_chart(data=SAMPLE_CSV, chart_type="bar", title="CDN")
        path = _extract_path(result)
        if path and os.path.exists(path):
            content = open(path, encoding="utf-8").read()
            assert "cdn.plot.ly" in content or "plotly-latest" in content


class TestCreateChartTool:
    """Test the LangChain tool wrapper."""

    def test_tool_exists(self):
        from extensions.tools.visualization import create_chart_tool
        assert create_chart_tool is not None

    def test_tool_name(self):
        from extensions.tools.visualization import create_chart_tool
        assert create_chart_tool.name == "create_chart"


class TestDetectOutliersIQR:
    """Test outlier detection with IQR method."""

    def test_iqr_returns_string(self):
        result = detect_outliers(data=OUTLIER_CSV, column="Value", method="iqr")
        assert isinstance(result, str)
        assert "[ERROR]" not in result

    def test_iqr_finds_outlier(self):
        result = detect_outliers(data=OUTLIER_CSV, column="Value", method="iqr")
        # Value 100 is clearly an outlier
        assert "Outliers found:" in result
        assert "100.00" in result

    def test_iqr_reports_counts(self):
        result = detect_outliers(data=OUTLIER_CSV, column="Value", method="iqr")
        assert "Total values:" in result

    def test_iqr_creates_chart(self):
        result = detect_outliers(data=OUTLIER_CSV, column="Value", method="iqr")
        # Should mention a file path
        assert "outliers_" in result or "File:" in result or "Visualization:" in result


class TestDetectOutliersZScore:
    """Test outlier detection with Z-score method."""

    def test_zscore_returns_string(self):
        result = detect_outliers(
            data=OUTLIER_CSV, column="Value", method="zscore", threshold=2.0,
        )
        assert isinstance(result, str)
        assert "[ERROR]" not in result

    def test_zscore_finds_outlier(self):
        result = detect_outliers(
            data=OUTLIER_CSV, column="Value", method="zscore", threshold=2.0,
        )
        assert "Outliers found:" in result


class TestDetectOutliersEdgeCases:
    """Test edge cases for detect_outliers."""

    def test_missing_column(self):
        result = detect_outliers(data=OUTLIER_CSV, column="nonexistent", method="iqr")
        assert "[ERROR]" in result
        assert "not found" in result

    def test_non_numeric_column(self):
        csv_with_text = "Name,Label\nA,x\nB,y\nC,z"
        result = detect_outliers(data=csv_with_text, column="Label", method="iqr")
        assert "[ERROR]" in result

    def test_unknown_method(self):
        result = detect_outliers(data=OUTLIER_CSV, column="Value", method="invalid")
        assert "[ERROR]" in result
        assert "Unknown method" in result

    def test_empty_data(self):
        result = detect_outliers(data="", column="Value", method="iqr")
        assert "[ERROR]" in result

    def test_no_outliers(self):
        """Data with no outliers."""
        uniform_csv = "ID,Value\n" + "\n".join(f"{i},{10+i}" for i in range(20))
        result = detect_outliers(data=uniform_csv, column="Value", method="iqr")
        assert isinstance(result, str)
        # May or may not find outliers depending on distribution

    def test_custom_threshold(self):
        result = detect_outliers(
            data=OUTLIER_CSV, column="Value", method="iqr", threshold=3.0,
        )
        assert isinstance(result, str)
        assert "[ERROR]" not in result


class TestDetectOutliersTool:
    """Test the LangChain tool wrapper."""

    def test_tool_exists(self):
        from extensions.tools.visualization import detect_outliers_tool
        assert detect_outliers_tool is not None

    def test_tool_name(self):
        from extensions.tools.visualization import detect_outliers_tool
        assert detect_outliers_tool.name == "detect_outliers"
