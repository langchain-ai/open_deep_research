import ast
from pathlib import Path

SOURCE_PATH = Path(__file__).parents[1] / "src" / "legacy" / "utils.py"


def _parse_source() -> ast.Module:
    return ast.parse(SOURCE_PATH.read_text())


def test_synchronous_http_requests_have_a_timeout():
    tree = _parse_source()
    timeout_value = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_HTTP_TIMEOUT"
            for target in node.targets
        )
    )
    assert isinstance(timeout_value, ast.Constant)
    assert timeout_value.value > 0

    request_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requests"
        and node.func.attr in {"get", "post"}
    ]
    assert len(request_calls) == 2

    for call in request_calls:
        timeout = next(keyword.value for keyword in call.keywords if keyword.arg == "timeout")
        assert isinstance(timeout, ast.Name)
        assert timeout.id == "DEFAULT_HTTP_TIMEOUT"


def test_perplexity_search_runs_off_the_event_loop():
    tree = _parse_source()
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "select_and_execute_search"
    )

    threaded_calls = [
        node.value
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "asyncio"
        and node.value.func.attr == "to_thread"
    ]
    assert len(threaded_calls) == 1
    assert isinstance(threaded_calls[0].args[0], ast.Name)
    assert threaded_calls[0].args[0].id == "perplexity_search"
