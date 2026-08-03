"""Regression tests for legacy cross-platform date formatting."""

import ast
import datetime
import types
import unittest
from pathlib import Path


class WindowsDateTime(datetime.datetime):
    """Simulate Windows rejecting POSIX-only strftime directives."""

    @classmethod
    def now(cls) -> "WindowsDateTime":
        return cls(2024, 1, 5)

    def strftime(self, date_format: str) -> str:
        if "%-" in date_format:
            raise ValueError("Invalid format string")
        return super().strftime(date_format)


def load_get_today_str():
    """Load only get_today_str without importing legacy's optional dependencies."""
    source_path = Path(__file__).resolve().parents[1] / "src" / "legacy" / "utils.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_today_str"
    )
    namespace = {
        "datetime": types.SimpleNamespace(datetime=WindowsDateTime),
    }
    exec(
        compile(ast.Module([function], type_ignores=[]), source_path, "exec"), namespace
    )
    return namespace["get_today_str"]


class LegacyDateFormatTest(unittest.TestCase):
    """Verify the legacy prompt date works with Windows strftime semantics."""

    def test_get_today_str_avoids_posix_only_directives(self) -> None:
        get_today_str = load_get_today_str()

        self.assertEqual(get_today_str(), "Fri Jan 5, 2024")


if __name__ == "__main__":
    unittest.main()
