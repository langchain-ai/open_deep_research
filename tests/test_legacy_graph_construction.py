import ast
from pathlib import Path

import pytest

DEPRECATED_SCHEMA_KWARGS = {"config_schema", "input", "output"}


@pytest.mark.parametrize(
    "module_name",
    ["graph.py", "multi_agent.py"],
)
def test_legacy_state_graphs_use_current_schema_kwargs(module_name) -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "legacy"
        / module_name
    )
    module = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "StateGraph"
        )
    ]

    assert calls
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert keywords.isdisjoint(DEPRECATED_SCHEMA_KWARGS)
