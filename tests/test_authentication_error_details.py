import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeAuth:
    class types:
        MinimalUserDict = dict

    class exceptions:
        HTTPException = FakeHTTPException


def _load_get_current_user(supabase):
    path = (
        Path(__file__).parents[1]
        / "src"
        / "security"
        / "auth.py"
    )
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "get_current_user"
    )
    function.decorator_list = []
    namespace = {
        "Any": Any,
        "Auth": FakeAuth,
        "asyncio": asyncio,
        "supabase": supabase,
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
    return namespace["get_current_user"]


def test_supabase_failure_does_not_expose_internal_details() -> None:
    class BrokenSupabaseAuth:
        def get_user(self, token):
            raise RuntimeError(
                "database host db.internal.example failed"
            )

    supabase = SimpleNamespace(auth=BrokenSupabaseAuth())
    get_current_user = _load_get_current_user(supabase)

    with pytest.raises(FakeHTTPException) as error:
        asyncio.run(get_current_user("Bearer token"))

    assert error.value.status_code == 401
    assert error.value.detail == "Authentication failed"
    assert "db.internal.example" not in error.value.detail


def test_deliberate_authentication_error_is_not_rewrapped() -> None:
    class EmptySupabaseAuth:
        def get_user(self, token):
            return SimpleNamespace(user=None)

    supabase = SimpleNamespace(auth=EmptySupabaseAuth())
    get_current_user = _load_get_current_user(supabase)

    with pytest.raises(FakeHTTPException) as error:
        asyncio.run(get_current_user("Bearer token"))

    assert error.value.status_code == 401
    assert error.value.detail == "Invalid token or user not found"
