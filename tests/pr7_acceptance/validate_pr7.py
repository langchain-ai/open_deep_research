from __future__ import annotations

import logging

from open_deep_research.configuration import (
    Configuration,
    MemoryMode,
    RagScope,
    SearchAPI,
)
from open_deep_research.deep_researcher import (
    _build_api_sources_and_ids,
    _build_effective_tools_config,
    _extract_token_usage_from_message,
    _is_local_rag_active,
    _is_long_term_memory_active,
    _is_session_memory_active,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DummyMessage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


def print_header(title: str) -> None:
    logger.info(f"\n=== {title} ===")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_memory_mode_gating() -> None:
    print_header("PR7 Memory Mode Gating")

    cfg_off = Configuration(memory_enabled=True, memory_mode=MemoryMode.OFF)
    assert_true(not _is_session_memory_active(cfg_off), "OFF should disable session memory")
    assert_true(not _is_long_term_memory_active(cfg_off), "OFF should disable long-term memory")

    cfg_session = Configuration(memory_enabled=True, memory_mode=MemoryMode.SESSION_ONLY)
    assert_true(_is_session_memory_active(cfg_session), "SESSION_ONLY should enable session memory")
    assert_true(not _is_long_term_memory_active(cfg_session), "SESSION_ONLY should disable long-term memory")

    cfg_long = Configuration(memory_enabled=True, memory_mode=MemoryMode.LONG_TERM_ONLY)
    assert_true(not _is_session_memory_active(cfg_long), "LONG_TERM_ONLY should disable session memory")
    assert_true(_is_long_term_memory_active(cfg_long), "LONG_TERM_ONLY should enable long-term memory")

    cfg_disabled_master = Configuration(memory_enabled=False, memory_mode=MemoryMode.BOTH)
    assert_true(not _is_session_memory_active(cfg_disabled_master), "memory_enabled=false should disable session memory")
    assert_true(not _is_long_term_memory_active(cfg_disabled_master), "memory_enabled=false should disable long-term memory")

    logger.info("PASS: memory_mode correctly gates session/long-term memory")


def validate_rag_scope_gating() -> None:
    print_header("PR7 RAG Scope Gating")

    base_config = {"configurable": {"search_api": SearchAPI.TAVILY.value, "rag_enabled": True}}

    cfg_disabled = Configuration(rag_enabled=True, rag_scope=RagScope.DISABLED)
    effective_disabled = _build_effective_tools_config(base_config, cfg_disabled)
    assert_true(effective_disabled["configurable"]["rag_enabled"] is False, "DISABLED should force rag_enabled=false")

    cfg_local = Configuration(rag_enabled=True, rag_scope=RagScope.LOCAL_ONLY)
    effective_local = _build_effective_tools_config(base_config, cfg_local)
    assert_true(
        effective_local["configurable"]["search_api"] == SearchAPI.NONE.value,
        "LOCAL_ONLY should force search_api=none",
    )
    assert_true(_is_local_rag_active(cfg_local), "LOCAL_ONLY should keep local RAG active")

    cfg_hybrid = Configuration(rag_enabled=True, rag_scope=RagScope.HYBRID)
    effective_hybrid = _build_effective_tools_config(base_config, cfg_hybrid)
    assert_true(
        effective_hybrid["configurable"]["search_api"] == SearchAPI.TAVILY.value,
        "HYBRID should keep configured search API",
    )
    assert_true(_is_local_rag_active(cfg_hybrid), "HYBRID should keep local RAG active")

    logger.info("PASS: rag_scope correctly gates tool availability")


def validate_api_sources_and_cost_metadata() -> None:
    print_header("PR7 API Sources/Cost Metadata")

    cfg = Configuration(memory_enabled=True)
    notes = [
        "--- LOCAL SOURCE 1 ---\nSource: D:/kb/finance.md\nUpdatedAt: 2026-03-18T08:00:00+00:00\nContent: Revenue growth: 12%",
        "URL: https://example.com/market\nSummary: Revenue growth: 10%",
    ]
    confirmed_prefs = ["language: 中文"]

    sources, citation_ids = _build_api_sources_and_ids(notes, confirmed_prefs, cfg)
    assert_true(len(sources) >= 3, "Expected fused sources from local/web/memory")
    assert_true(citation_ids[0] == "SRC-001", "Citation IDs should start from SRC-001")
    assert_true(all(source.get("id", "").startswith("SRC-") for source in sources), "Source entries must carry SRC IDs")

    usage_from_metadata = _extract_token_usage_from_message(
        DummyMessage(usage_metadata={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16})
    )
    assert_true(usage_from_metadata["total_tokens"] == 16, "usage_metadata tokens should be extracted")

    usage_from_response = _extract_token_usage_from_message(
        DummyMessage(response_metadata={"usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}})
    )
    assert_true(usage_from_response["total_tokens"] == 7, "response_metadata usage should be extracted")

    logger.info("PASS: API source list and cost token usage metadata are available")


def main() -> int:
    validate_memory_mode_gating()
    validate_rag_scope_gating()
    validate_api_sources_and_cost_metadata()
    logger.info("\nAll PR-7 acceptance checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
