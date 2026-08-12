"""Regression tests for store namespace authorization."""

import ast
import asyncio
import types
import unittest
from pathlib import Path


class HTTPException(Exception):
    """Minimal stand-in for LangGraph's HTTP exception."""

    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class StudioUser:
    """Stand-in for the Studio user type."""


def load_authorize_store():
    """Load the authorization handler without importing external auth clients."""
    source_path = Path(__file__).resolve().parents[1] / "src" / "security" / "auth.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "authorize_store"
    )
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None

    auth = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(HTTPException=HTTPException),
    )
    namespace = {"Auth": auth, "StudioUser": StudioUser}
    exec(
        compile(ast.Module([function], type_ignores=[]), source_path, "exec"), namespace
    )
    return function, namespace["authorize_store"]


class StoreAuthorizationTest(unittest.TestCase):
    """Verify store ownership checks survive optimized Python execution."""

    def setUp(self) -> None:
        self.function, self.authorize_store = load_authorize_store()
        self.context = types.SimpleNamespace(
            user=types.SimpleNamespace(identity="alice"),
        )

    def test_owner_namespace_is_allowed(self) -> None:
        result = asyncio.run(
            self.authorize_store(self.context, {"namespace": ("alice", "notes")})
        )

        self.assertIsNone(result)

    def test_cross_user_namespace_is_forbidden(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                self.authorize_store(self.context, {"namespace": ("bob", "notes")})
            )

        self.assertEqual(raised.exception.status_code, 403)

    def test_empty_namespace_is_forbidden(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self.authorize_store(self.context, {"namespace": ()}))

        self.assertEqual(raised.exception.status_code, 403)

    def test_handler_does_not_use_optimized_away_assertions(self) -> None:
        self.assertFalse(
            any(isinstance(node, ast.Assert) for node in ast.walk(self.function))
        )


if __name__ == "__main__":
    unittest.main()
