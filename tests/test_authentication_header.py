"""Regression tests for authentication header validation."""

import ast
import asyncio
import types
import unittest
from pathlib import Path
from typing import Any


class HTTPException(Exception):
    """Minimal stand-in for LangGraph's HTTP exception."""

    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeSupabase:
    """Return a valid user for any token presented to the client."""

    def __init__(self):
        self.auth = types.SimpleNamespace(
            get_user=lambda token: types.SimpleNamespace(
                user=types.SimpleNamespace(id=f"user:{token}"),
            )
        )


def load_get_current_user(*, optimize: int):
    """Load the auth handler with the requested Python optimization level."""
    source_path = Path(__file__).resolve().parents[1] / "src" / "security" / "auth.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_current_user"
    )
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None

    auth = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(HTTPException=HTTPException),
    )
    namespace = {
        "Any": Any,
        "Auth": auth,
        "asyncio": asyncio,
        "supabase": FakeSupabase(),
    }
    exec(
        compile(
            ast.Module([function], type_ignores=[]),
            source_path,
            "exec",
            optimize=optimize,
        ),
        namespace,
    )
    return function, namespace["get_current_user"]


class AuthenticationHeaderTest(unittest.TestCase):
    """Verify authentication schemes are enforced without assertions."""

    def test_bearer_token_is_accepted(self) -> None:
        _, authenticate = load_get_current_user(optimize=0)

        user = asyncio.run(authenticate("Bearer valid-token"))

        self.assertEqual(user, {"identity": "user:valid-token"})

    def test_basic_scheme_is_rejected_normally(self) -> None:
        _, authenticate = load_get_current_user(optimize=0)

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(authenticate("Basic valid-token"))

        self.assertEqual(raised.exception.status_code, 401)

    def test_basic_scheme_is_rejected_with_optimization(self) -> None:
        _, authenticate = load_get_current_user(optimize=1)

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(authenticate("Basic valid-token"))

        self.assertEqual(raised.exception.status_code, 401)

    def test_handler_does_not_use_optimized_away_assertions(self) -> None:
        function, _ = load_get_current_user(optimize=0)

        self.assertFalse(
            any(isinstance(node, ast.Assert) for node in ast.walk(function))
        )


if __name__ == "__main__":
    unittest.main()
