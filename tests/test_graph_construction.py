"""Regression tests for StateGraph construction in the deep researcher graph."""

import ast
import pathlib
import unittest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "open_deep_research"
    / "deep_researcher.py"
).read_text()

DEPRECATED_KWARGS = {"config_schema", "input", "output"}


class StateGraphConstructionTest(unittest.TestCase):
    """Verify StateGraph is built with the non-deprecated LangGraph API."""

    def _state_graph_calls(self) -> list[ast.Call]:
        tree = ast.parse(SOURCE)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "StateGraph":
                    calls.append(node)
        return calls

    def test_state_graph_is_used(self) -> None:
        self.assertEqual(len(self._state_graph_calls()), 3)

    def test_no_deprecated_state_graph_kwargs(self) -> None:
        for call in self._state_graph_calls():
            kwargs = {kw.arg for kw in call.keywords if kw.arg is not None}
            offending = kwargs & DEPRECATED_KWARGS
            self.assertFalse(
                offending,
                msg=f"StateGraph called with deprecated kwargs: {sorted(offending)}",
            )


if __name__ == "__main__":
    unittest.main()
