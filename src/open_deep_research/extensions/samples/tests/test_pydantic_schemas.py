"""Unit tests for Pydantic schemas: ExtractedTable, ExtractedDataset, tool schemas."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestExtractedTable:
    """Test ExtractedTable Pydantic model."""

    def test_valid_table(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        table = ExtractedTable(
            table_name="GDP by Country",
            headers=["Country", "GDP_Trillion"],
            rows=[["USA", "25.5"], ["China", "17.9"]],
        )
        assert table.table_name == "GDP by Country"
        assert table.headers == ["Country", "GDP_Trillion"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["USA", "25.5"]

    def test_empty_rows(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        table = ExtractedTable(
            table_name="Empty",
            headers=["A", "B"],
            rows=[],
        )
        assert table.rows == []

    def test_single_row(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        table = ExtractedTable(
            table_name="Single",
            headers=["X", "Y"],
            rows=[["1", "2"]],
        )
        assert len(table.rows) == 1

    def test_many_columns(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        headers = [f"col_{i}" for i in range(20)]
        rows = [[str(i * j) for j in range(20)] for i in range(5)]
        table = ExtractedTable(table_name="Wide", headers=headers, rows=rows)
        assert len(table.headers) == 20
        assert len(table.rows) == 5

    def test_serialization_roundtrip(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        table = ExtractedTable(
            table_name="Test",
            headers=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
        )
        d = table.model_dump()
        assert d["table_name"] == "Test"
        assert d["headers"] == ["A", "B"]
        assert d["rows"] == [["1", "2"], ["3", "4"]]

        # Reconstruct from dict
        table2 = ExtractedTable(**d)
        assert table2 == table

    def test_json_roundtrip(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        table = ExtractedTable(
            table_name="JSON Test",
            headers=["Name", "Value"],
            rows=[["alpha", "100"]],
        )
        json_str = table.model_dump_json()
        table2 = ExtractedTable.model_validate_json(json_str)
        assert table2.table_name == "JSON Test"
        assert table2.rows == [["alpha", "100"]]

    def test_missing_table_name_raises(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ExtractedTable(headers=["A"], rows=[["1"]])

    def test_missing_headers_raises(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ExtractedTable(table_name="X", rows=[["1"]])

    def test_missing_rows_raises(self):
        from extensions.models.extracted_data_schema import ExtractedTable
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ExtractedTable(table_name="X", headers=["A"])


class TestExtractedDataset:
    """Test ExtractedDataset Pydantic model."""

    def test_empty_dataset(self):
        from extensions.models.extracted_data_schema import ExtractedDataset
        ds = ExtractedDataset()
        assert ds.tables == []

    def test_dataset_with_tables(self):
        from extensions.models.extracted_data_schema import ExtractedTable, ExtractedDataset
        t1 = ExtractedTable(table_name="T1", headers=["A", "B"], rows=[["1", "2"]])
        t2 = ExtractedTable(table_name="T2", headers=["X", "Y", "Z"], rows=[["a", "b", "c"]])
        ds = ExtractedDataset(tables=[t1, t2])
        assert len(ds.tables) == 2
        assert ds.tables[0].table_name == "T1"
        assert ds.tables[1].table_name == "T2"

    def test_dataset_serialization(self):
        from extensions.models.extracted_data_schema import ExtractedTable, ExtractedDataset
        t1 = ExtractedTable(table_name="T1", headers=["A"], rows=[["x"]])
        ds = ExtractedDataset(tables=[t1])
        d = ds.model_dump()
        assert len(d["tables"]) == 1
        assert d["tables"][0]["table_name"] == "T1"

    def test_dataset_json_roundtrip(self):
        from extensions.models.extracted_data_schema import ExtractedTable, ExtractedDataset
        t1 = ExtractedTable(table_name="T1", headers=["C1", "C2"], rows=[["v1", "v2"]])
        ds = ExtractedDataset(tables=[t1])
        json_str = ds.model_dump_json()
        ds2 = ExtractedDataset.model_validate_json(json_str)
        assert len(ds2.tables) == 1
        assert ds2.tables[0].headers == ["C1", "C2"]

    def test_dataset_default_factory(self):
        """Verify default_factory=list creates independent instances."""
        from extensions.models.extracted_data_schema import ExtractedDataset
        ds1 = ExtractedDataset()
        ds2 = ExtractedDataset()
        assert ds1.tables is not ds2.tables


class TestToolSchemas:
    """Test tool input schemas from tool_schemas.py."""

    def test_add_input(self):
        from extensions.models.tool_schemas import AddInput
        inp = AddInput(a=1.5, b=2.5)
        assert inp.a == 1.5
        assert inp.b == 2.5

    def test_divide_input(self):
        from extensions.models.tool_schemas import DivideInput
        inp = DivideInput(a=10.0, b=3.0)
        assert inp.a == 10.0

    def test_calculate_input(self):
        from extensions.models.tool_schemas import CalculateInput
        inp = CalculateInput(expression="sqrt(16)")
        assert inp.expression == "sqrt(16)"

    def test_data_profiling_input_defaults(self):
        from extensions.models.tool_schemas import DataProfilingInput
        inp = DataProfilingInput(data="some csv")
        assert inp.analysis_type == "comprehensive"

    def test_data_extraction_input_defaults(self):
        from extensions.models.tool_schemas import DataExtractionInput
        inp = DataExtractionInput(text="some text")
        assert inp.format == "json"

    def test_plotly_input_defaults(self):
        from extensions.models.tool_schemas import PlotlyVisualizationInput
        inp = PlotlyVisualizationInput(data="csv", chart_type="bar")
        assert inp.title == "Chart"
        assert inp.x_column == ""
        assert inp.y_column == ""

    def test_outlier_detection_input_defaults(self):
        from extensions.models.tool_schemas import OutlierDetectionInput
        inp = OutlierDetectionInput(data="csv", column="val")
        assert inp.method == "iqr"
        assert inp.threshold == 1.5

    def test_chart_explanation(self):
        from extensions.models.tool_schemas import ChartExplanation
        ce = ChartExplanation(
            chart_path="/tmp/chart.html",
            title="Bar Chart",
            explanation="Shows distribution",
        )
        assert ce.chart_path == "/tmp/chart.html"

    def test_chart_explanations_collection(self):
        from extensions.models.tool_schemas import ChartExplanation, ChartExplanations
        ce = ChartExplanation(chart_path="a.html", title="T", explanation="E")
        coll = ChartExplanations(charts=[ce])
        assert len(coll.charts) == 1

    def test_extracted_chart_paths_default(self):
        from extensions.models.tool_schemas import ExtractedChartPaths
        ecp = ExtractedChartPaths()
        assert ecp.paths == []

    def test_extracted_chart_paths_with_data(self):
        from extensions.models.tool_schemas import ExtractedChartPaths
        ecp = ExtractedChartPaths(paths=["a.html", "b.html"])
        assert len(ecp.paths) == 2
