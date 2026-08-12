import ast
import asyncio
from pathlib import Path

import pytest


def _load_search_dispatcher():
    path = Path(__file__).parents[1] / "src" / "legacy" / "utils.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "select_and_execute_search"
    )
    namespace = {}
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
    return namespace["select_and_execute_search"]


def test_none_search_returns_empty_context() -> None:
    dispatcher = _load_search_dispatcher()

    result = asyncio.run(
        dispatcher(
            search_api="none",
            query_list=["unused query"],
            params_to_pass={},
        )
    )

    assert result == ""


def test_unknown_search_api_still_fails() -> None:
    dispatcher = _load_search_dispatcher()

    with pytest.raises(ValueError, match="Unsupported search API: unknown"):
        asyncio.run(
            dispatcher(
                search_api="unknown",
                query_list=[],
                params_to_pass={},
            )
        )
