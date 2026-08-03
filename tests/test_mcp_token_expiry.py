import ast
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


class FakeStore:
    def __init__(self, item):
        self.item = item
        self.deleted = []

    async def aget(self, namespace, key):
        return self.item

    async def adelete(self, namespace, key):
        self.deleted.append((namespace, key))


def _load_get_tokens(store):
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
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "get_tokens"
    )
    namespace = {
        "RunnableConfig": dict,
        "get_store": lambda: store,
        "datetime": datetime,
        "timedelta": timedelta,
        "timezone": timezone,
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
    return namespace["get_tokens"]


def _config():
    return {
        "configurable": {"thread_id": "thread"},
        "metadata": {"owner": "user"},
    }


def test_token_without_expiry_is_returned() -> None:
    value = {"access_token": "token"}
    item = SimpleNamespace(
        value=value,
        created_at=datetime.now(timezone.utc),
    )
    store = FakeStore(item)

    result = asyncio.run(_load_get_tokens(store)(_config()))

    assert result == value
    assert store.deleted == []


def test_expired_token_is_still_deleted() -> None:
    item = SimpleNamespace(
        value={"access_token": "token", "expires_in": 60},
        created_at=datetime.now(timezone.utc) - timedelta(seconds=120),
    )
    store = FakeStore(item)

    result = asyncio.run(_load_get_tokens(store)(_config()))

    assert result is None
    assert store.deleted == [(("user", "tokens"), "data")]
