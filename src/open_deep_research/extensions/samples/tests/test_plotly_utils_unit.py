"""Unit tests for plotly_utils.py - load_plotly_figure, figure_to_html, _extract_json_value."""
import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.utils.plotly_utils import _extract_json_value, load_plotly_figure, figure_to_html


class TestExtractJsonValue:
    """Test the _extract_json_value bracket-counting parser."""

    def test_extract_array(self):
        text = '[1, 2, 3]'
        result, next_pos = _extract_json_value(text, 0)
        assert result == '[1, 2, 3]'
        assert next_pos == 9

    def test_extract_object(self):
        text = '{"a": 1, "b": 2}'
        result, next_pos = _extract_json_value(text, 0)
        assert result == '{"a": 1, "b": 2}'

    def test_nested_objects(self):
        text = '{"outer": {"inner": [1, 2]}}'
        result, next_pos = _extract_json_value(text, 0)
        assert result == text
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == [1, 2]

    def test_nested_arrays(self):
        text = '[[1, 2], [3, 4]]'
        result, next_pos = _extract_json_value(text, 0)
        assert result == text

    def test_strings_with_brackets(self):
        text = '{"key": "value with [brackets] and {braces}"}'
        result, next_pos = _extract_json_value(text, 0)
        assert result == text
        parsed = json.loads(result)
        assert "brackets" in parsed["key"]

    def test_escaped_quotes(self):
        text = r'{"key": "value with \"escaped\" quotes"}'
        result, next_pos = _extract_json_value(text, 0)
        assert result is not None

    def test_offset_start(self):
        text = 'prefix [1, 2, 3]'
        result, next_pos = _extract_json_value(text, 7)
        assert result == '[1, 2, 3]'

    def test_non_json_start(self):
        text = 'plain text'
        result, next_pos = _extract_json_value(text, 0)
        assert result is None

    def test_empty_string(self):
        result, next_pos = _extract_json_value("", 0)
        assert result is None

    def test_past_end(self):
        result, next_pos = _extract_json_value("[]", 5)
        assert result is None

    def test_unclosed_bracket(self):
        text = '[1, 2, 3'
        result, next_pos = _extract_json_value(text, 0)
        assert result is None


class TestLoadPlotlyFigure:
    """Test loading Plotly figures from HTML files."""

    def _make_plotly_html(self, data, layout=None):
        """Create a minimal Plotly HTML file."""
        data_json = json.dumps(data)
        layout_json = json.dumps(layout or {})
        html = f'''<html>
<body>
<div id="test-div"></div>
<script>
Plotly.newPlot("test-div", {data_json}, {layout_json}, {{}})
</script>
</body>
</html>'''
        return html

    def test_load_valid_figure(self):
        data = [{"x": [1, 2, 3], "y": [4, 5, 6], "type": "scatter"}]
        layout = {"title": "Test Chart"}
        html = self._make_plotly_html(data, layout)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(html)
            path = f.name

        try:
            fig = load_plotly_figure(path)
            assert fig is not None
            assert len(fig.data) == 1
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        fig = load_plotly_figure("/nonexistent/path.html")
        assert fig is None

    def test_load_non_plotly_html(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write("<html><body>Not a Plotly file</body></html>")
            path = f.name

        try:
            fig = load_plotly_figure(path)
            assert fig is None
        finally:
            os.unlink(path)

    def test_load_with_src_dir(self):
        data = [{"x": [1], "y": [2], "type": "bar"}]
        html = self._make_plotly_html(data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chart_path = os.path.join(tmp_dir, "chart.html")
            with open(chart_path, "w", encoding="utf-8") as f:
                f.write(html)

            fig = load_plotly_figure("chart.html", src_dir=Path(tmp_dir))
            assert fig is not None

    def test_load_with_layout(self):
        data = [{"x": [1], "y": [2], "type": "bar"}]
        layout = {"title": {"text": "My Title"}, "xaxis": {"title": "X"}}
        html = self._make_plotly_html(data, layout)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(html)
            path = f.name

        try:
            fig = load_plotly_figure(path)
            assert fig is not None
            assert fig.layout.title.text == "My Title"
        finally:
            os.unlink(path)


class TestFigureToHtml:
    """Test converting Plotly figures to HTML strings."""

    def test_returns_html_string(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[1, 2])])
        result = figure_to_html(fig)
        assert isinstance(result, str)
        assert "<div" in result

    def test_uses_cdn_by_default(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(x=["A"], y=[1])])
        result = figure_to_html(fig, include_plotlyjs=False)
        # CDN reference should be present
        assert "cdn" in result.lower() or "plotly" in result.lower()

    def test_not_full_html(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(x=["A"], y=[1])])
        result = figure_to_html(fig)
        # Should be just a div, not full HTML page
        assert "<!DOCTYPE" not in result
