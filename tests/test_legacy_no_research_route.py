import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Literal


class FakeCommand:
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, goto, update=None):
        self.goto = goto
        self.update = update


class FakeSend:
    def __init__(self, node, arg):
        self.node = node
        self.arg = arg


def _load_human_feedback(feedback):
    path = Path(__file__).parents[1] / "src" / "legacy" / "graph.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "human_feedback"
    )
    namespace = {
        "ReportState": dict,
        "RunnableConfig": dict,
        "Command": FakeCommand,
        "Literal": Literal,
        "Send": FakeSend,
        "interrupt": lambda message: feedback,
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
    return namespace["human_feedback"]


def _section(name: str, research: bool):
    return SimpleNamespace(
        name=name,
        description=f"{name} description",
        research=research,
    )


def test_approved_plan_without_research_starts_final_section_writers() -> None:
    human_feedback = _load_human_feedback(True)
    sections = [
        _section("Introduction", False),
        _section("Conclusion", False),
    ]

    result = human_feedback(
        {"topic": "topic", "sections": sections},
        {},
    )

    assert [send.node for send in result.goto] == [
        "write_final_sections",
        "write_final_sections",
    ]
    assert [send.arg["section"] for send in result.goto] == sections
    assert all(
        send.arg["report_sections_from_research"] == ""
        for send in result.goto
    )


def test_mixed_plan_still_starts_only_research_sections() -> None:
    human_feedback = _load_human_feedback(True)
    researched = _section("Body", True)
    final = _section("Conclusion", False)

    result = human_feedback(
        {"topic": "topic", "sections": [researched, final]},
        {},
    )

    assert len(result.goto) == 1
    assert result.goto[0].node == "build_section_with_web_research"
    assert result.goto[0].arg["section"] is researched


def test_string_feedback_still_regenerates_plan() -> None:
    human_feedback = _load_human_feedback("Add a comparison")

    result = human_feedback(
        {
            "topic": "topic",
            "sections": [_section("Body", True)],
        },
        {},
    )

    assert result.goto == "generate_report_plan"
    assert result.update == {
        "feedback_on_report_plan": ["Add a comparison"]
    }
