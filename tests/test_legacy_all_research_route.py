import ast
from pathlib import Path
from types import SimpleNamespace


class FakeSend:
    def __init__(self, node, arg):
        self.node = node
        self.arg = arg


def _graph_module():
    path = Path(__file__).parents[1] / "src" / "legacy" / "graph.py"
    return path, ast.parse(path.read_text())


def _load_route():
    path, module = _graph_module()
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "initiate_final_section_writing"
    )
    namespace = {
        "ReportState": dict,
        "Send": FakeSend,
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
    return namespace["initiate_final_section_writing"]


def test_all_researched_sections_route_to_report_compilation() -> None:
    route = _load_route()
    state = {
        "topic": "topic",
        "sections": [SimpleNamespace(research=True)],
        "report_sections_from_research": "researched body",
    }

    assert route(state) == "compile_final_report"


def test_non_research_sections_still_route_to_writers() -> None:
    route = _load_route()
    researched = SimpleNamespace(research=True)
    final = SimpleNamespace(research=False)
    state = {
        "topic": "topic",
        "sections": [researched, final],
        "report_sections_from_research": "researched body",
    }

    result = route(state)

    assert len(result) == 1
    assert result[0].node == "write_final_sections"
    assert result[0].arg["section"] is final


def test_conditional_edge_allows_direct_report_compilation() -> None:
    _, module = _graph_module()
    edge = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_conditional_edges"
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "gather_completed_sections"
    )

    path_map = ast.literal_eval(edge.args[2])

    assert "compile_final_report" in path_map
