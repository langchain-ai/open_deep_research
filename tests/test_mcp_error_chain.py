import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeMcpError(Exception):
    def __init__(self, error):
        super().__init__("MCP request failed")
        self.error = error


class FakeToolException(Exception):
    pass


def _load_wrapper():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "open_deep_research"
        / "utils.py"
    )
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "wrap_mcp_authenticate_tool"
    )
    namespace = {
        "StructuredTool": object,
        "McpError": FakeMcpError,
        "ToolException": FakeToolException,
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
    return namespace["wrap_mcp_authenticate_tool"]


def test_wrapped_mcp_interaction_error_is_translated() -> None:
    async def call_tool(**kwargs):
        error = SimpleNamespace(
            code=-32003,
            data={
                "message": {"text": "Complete sign-in"},
                "url": "https://auth.example",
            },
        )
        try:
            raise FakeMcpError(error)
        except FakeMcpError as exc:
            raise RuntimeError("adapter wrapper failed") from exc

    tool = SimpleNamespace(coroutine=call_tool)
    wrapped = _load_wrapper()(tool)

    with pytest.raises(
        FakeToolException,
        match="Complete sign-in https://auth.example",
    ):
        asyncio.run(wrapped.coroutine())


def test_non_mcp_cause_chain_is_preserved() -> None:
    async def call_tool(**kwargs):
        try:
            raise KeyError("inner")
        except KeyError as exc:
            raise ValueError("outer") from exc

    tool = SimpleNamespace(coroutine=call_tool)
    wrapped = _load_wrapper()(tool)

    with pytest.raises(ValueError, match="outer"):
        asyncio.run(wrapped.coroutine())
