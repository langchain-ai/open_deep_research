from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

from open_deep_research.configuration import Configuration, MemoryWritePolicy
from open_deep_research.deep_researcher import (
    _load_long_term_preferences,
    _process_long_term_memory_turn,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    logger.info(f"\n=== {title} ===")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class _StoreRecord:
    value: dict[str, Any]
    created_at: datetime = datetime.now(timezone.utc)


class _FakeStore:
    def __init__(self) -> None:
        self._data: dict[tuple[tuple[str, ...], str], _StoreRecord] = {}

    async def aget(self, namespace: tuple[str, ...], key: str) -> _StoreRecord | None:
        return self._data.get((namespace, key))

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        self._data[(namespace, key)] = _StoreRecord(value=value, created_at=datetime.now(timezone.utc))


async def validate_no_write_without_confirmation() -> tuple[list[dict[str, str]], _FakeStore, Configuration]:
    print_header("PR5 Explicit Confirmation Gate")

    store = _FakeStore()
    configurable = Configuration(
        memory_enabled=True,
        memory_write_policy=MemoryWritePolicy.EXPLICIT_CONFIRMATION,
        memory_namespace_prefix="memory-pr5-test",
        memory_max_candidates_per_turn=5,
    )
    config = {"metadata": {"owner": "owner-a"}}

    update = await _process_long_term_memory_turn(
        messages=[HumanMessage(content="请用中文简洁回答，关注开源方案，不要表格")],
        existing_pending_candidates=[],
        existing_confirmed_preferences=[],
        configurable=configurable,
        config=config,
        store_override=store,
    )

    pending = update["memory_candidates_pending_confirmation"]["value"]
    assert_true(len(pending) > 0, "Expected pending candidates to be generated")

    loaded = await _load_long_term_preferences(
        configurable=configurable,
        config=config,
        store_override=store,
    )
    assert_true(loaded == [], "Expected no persisted memory before explicit confirmation")

    logger.info("PASS: no long-term write occurs before explicit user confirmation")
    return pending, store, configurable


async def validate_write_then_read_next_session(
    pending_candidates: list[dict[str, str]],
    store: _FakeStore,
    configurable: Configuration,
) -> None:
    print_header("PR5 Confirmed Write And Next-Session Read")

    config = {"metadata": {"owner": "owner-a"}}

    update = await _process_long_term_memory_turn(
        messages=[HumanMessage(content="确认记忆")],
        existing_pending_candidates=pending_candidates,
        existing_confirmed_preferences=[],
        configurable=configurable,
        config=config,
        store_override=store,
    )

    pending_after_confirm = update["memory_candidates_pending_confirmation"]["value"]
    assert_true(pending_after_confirm == [], "Expected pending candidates to clear after confirmation")

    loaded_next_session = await _load_long_term_preferences(
        configurable=configurable,
        config=config,
        store_override=store,
    )
    assert_true(len(loaded_next_session) > 0, "Expected persisted long-term preferences after confirmation")
    assert_true(
        any(item.get("kind") == "language" and item.get("value") == "中文" for item in loaded_next_session),
        "Expected confirmed language preference to be persisted",
    )

    logger.info("PASS: confirmed preferences persist and are readable in subsequent sessions")


async def validate_owner_isolation(store: _FakeStore, configurable: Configuration) -> None:
    print_header("PR5 Owner Isolation")

    config_owner_a = {"metadata": {"owner": "owner-a"}}
    config_owner_b = {"metadata": {"owner": "owner-b"}}

    loaded_a = await _load_long_term_preferences(
        configurable=configurable,
        config=config_owner_a,
        store_override=store,
    )
    loaded_b = await _load_long_term_preferences(
        configurable=configurable,
        config=config_owner_b,
        store_override=store,
    )

    assert_true(len(loaded_a) > 0, "Expected owner-a to have persisted long-term memory")
    assert_true(loaded_b == [], "Expected owner-b to be fully isolated from owner-a memory")

    logger.info("PASS: long-term memory is isolated by owner namespace")


async def _run() -> int:
    pending, store, configurable = await validate_no_write_without_confirmation()
    await validate_write_then_read_next_session(pending, store, configurable)
    await validate_owner_isolation(store, configurable)
    logger.info("\nAll PR-5 acceptance checks finished.")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
