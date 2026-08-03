import ast
from pathlib import Path
from types import SimpleNamespace


class FakeConfiguration:
    include_source_str = False


class ConfigurationFactory:
    @classmethod
    def from_runnable_config(cls, config):
        return FakeConfiguration()


def _load_compile_final_report():
    path = Path(__file__).parents[1] / "src" / "legacy" / "graph.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compile_final_report"
    )
    namespace = {
        "ReportState": dict,
        "RunnableConfig": dict,
        "Configuration": ConfigurationFactory,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[]),
            ),
            filename=str(path),
            mode="exec",
        ),
        namespace,
    )
    return namespace["compile_final_report"]


def test_final_report_compilation_does_not_mutate_sections() -> None:
    first = SimpleNamespace(name="First", content="first placeholder")
    second = SimpleNamespace(name="Second", content="second placeholder")
    state = {
        "sections": [first, second],
        "completed_sections": [
            SimpleNamespace(name="Second", content="second researched"),
            SimpleNamespace(name="First", content="first researched"),
        ],
    }

    result = _load_compile_final_report()(state, {})

    assert result == {
        "final_report": "first researched\n\nsecond researched"
    }
    assert first.content == "first placeholder"
    assert second.content == "second placeholder"
