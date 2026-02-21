"""Unit tests for data_extraction.py - regex-based extract_data function."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.tools.data_extraction import extract_data


# ── Test Data ──

MARKDOWN_TABLE = """\
| Country | GDP | Growth |
|---------|-----|--------|
| USA | 25.5 | 2.1 |
| China | 17.9 | 5.2 |
| India | 3.7 | 6.8 |
"""

KEY_VALUE_TEXT = """\
Name: John Smith
Age: 30
City: New York
Occupation: Engineer
"""

BULLETED_LIST = """\
- Apple
- Banana
- Cherry
- Dragonfruit
"""

NUMBERED_LIST = """\
1. First item
2. Second item
3. Third item
"""

PLAIN_TEXT = "This is just a plain paragraph with no structured data at all."


class TestExtractDataMarkdownTable:
    """Test Method 1: Markdown table extraction."""

    def test_markdown_table_json(self):
        result = extract_data(MARKDOWN_TABLE, format="json")
        assert isinstance(result, str)
        # Should not be an error
        assert "[ERROR]" not in result

    def test_markdown_table_csv(self):
        result = extract_data(MARKDOWN_TABLE, format="csv")
        assert isinstance(result, str)
        # CSV should have comma-separated values
        if "[ERROR]" not in result:
            lines = result.strip().split("\n")
            assert len(lines) >= 1  # At least header line

    def test_markdown_table_table_format(self):
        result = extract_data(MARKDOWN_TABLE, format="table")
        assert isinstance(result, str)
        if "[ERROR]" not in result:
            assert "|" in result or "-" in result

    def test_markdown_table_returns_rows(self):
        import json
        result = extract_data(MARKDOWN_TABLE, format="json")
        if "[ERROR]" not in result:
            data = json.loads(result)
            assert isinstance(data, list)


class TestExtractDataKeyValue:
    """Test Method 2: Key-value pair extraction."""

    def test_kv_json(self):
        result = extract_data(KEY_VALUE_TEXT, format="json")
        assert isinstance(result, str)
        assert "[ERROR]" not in result

    def test_kv_csv(self):
        result = extract_data(KEY_VALUE_TEXT, format="csv")
        assert isinstance(result, str)
        assert "[ERROR]" not in result

    def test_kv_contains_values(self):
        import json
        result = extract_data(KEY_VALUE_TEXT, format="json")
        if "[ERROR]" not in result:
            data = json.loads(result)
            assert isinstance(data, list)
            assert len(data) >= 1


class TestExtractDataBulletedList:
    """Test Method 3: Bulleted/numbered list extraction."""

    def test_bulleted_list_json(self):
        result = extract_data(BULLETED_LIST, format="json")
        assert "[ERROR]" not in result

    def test_bulleted_list_items(self):
        import json
        result = extract_data(BULLETED_LIST, format="json")
        if "[ERROR]" not in result:
            data = json.loads(result)
            assert isinstance(data, list)
            assert len(data) >= 3

    def test_numbered_list_json(self):
        result = extract_data(NUMBERED_LIST, format="json")
        assert "[ERROR]" not in result

    def test_numbered_list_items(self):
        import json
        result = extract_data(NUMBERED_LIST, format="json")
        if "[ERROR]" not in result:
            data = json.loads(result)
            assert isinstance(data, list)
            assert len(data) >= 2


class TestExtractDataEdgeCases:
    """Test edge cases and error handling."""

    def test_plain_text_returns_error(self):
        result = extract_data(PLAIN_TEXT, format="json")
        assert "[ERROR]" in result
        assert "No structured data" in result

    def test_empty_string(self):
        result = extract_data("", format="json")
        assert "[ERROR]" in result

    def test_whitespace_only(self):
        result = extract_data("   \n\n  ", format="json")
        assert "[ERROR]" in result

    def test_default_format_is_json(self):
        result = extract_data(KEY_VALUE_TEXT)
        assert isinstance(result, str)
        # Default should be json format
        if "[ERROR]" not in result:
            import json
            json.loads(result)  # Should not raise

    def test_unknown_format_falls_back_to_json(self):
        result = extract_data(KEY_VALUE_TEXT, format="xml")
        assert isinstance(result, str)
        # Should fall through to the final json.dumps at end of function
        if "[ERROR]" not in result:
            import json
            json.loads(result)

    def test_csv_format_has_commas(self):
        result = extract_data(KEY_VALUE_TEXT, format="csv")
        if "[ERROR]" not in result:
            assert "," in result

    def test_table_format_has_separators(self):
        result = extract_data(KEY_VALUE_TEXT, format="table")
        if "[ERROR]" not in result:
            assert "-" in result


class TestExtractDataTool:
    """Test the LangChain tool wrapper."""

    def test_tool_exists(self):
        from extensions.tools.data_extraction import extract_data_tool
        assert extract_data_tool is not None

    def test_tool_name(self):
        from extensions.tools.data_extraction import extract_data_tool
        assert extract_data_tool.name == "extract_data"

    def test_tool_has_description(self):
        from extensions.tools.data_extraction import extract_data_tool
        assert len(extract_data_tool.description) > 0
