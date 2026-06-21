from __future__ import annotations

import logging

from open_deep_research.configuration import Configuration, EvidencePriorityStrategy
from open_deep_research.deep_researcher import _build_traceable_evidence_appendix

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    logger.info(f"\n=== {title} ===")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_three_channel_fusion_and_citations() -> None:
    print_header("PR6 Three-Channel Fusion & Citation Traceability")

    cfg = Configuration(
        memory_enabled=True,
        evidence_priority_strategy=EvidencePriorityStrategy.LOCAL_FIRST,
    )

    notes = [
        "--- LOCAL SOURCE 1 ---\nSource: D:/kb/alpha.md\nUpdatedAt: 2026-03-18T10:00:00+00:00\nContent: Battery life: 12h",
        "URL: https://example.com/review-alpha\nSummary: Battery life: 10h",
    ]
    confirmed_prefs = ["language: 中文", "style: 简洁"]

    appendix, citations = _build_traceable_evidence_appendix(notes, confirmed_prefs, cfg)

    assert_true("<UnifiedEvidence>" in appendix, "Expected unified evidence section")
    assert_true("channel=local" in appendix, "Expected local channel evidence")
    assert_true("channel=web" in appendix, "Expected web channel evidence")
    assert_true("channel=memory" in appendix, "Expected memory channel evidence")
    assert_true(len(citations) >= 3, "Expected citation IDs for fused evidence")
    assert_true("[SRC-001]" in appendix, "Expected stable citation marker")

    logger.info("PASS: three-channel fusion emits traceable citation mapping")


def validate_priority_strategy_local_first() -> None:
    print_header("PR6 Priority Strategy")

    cfg = Configuration(
        memory_enabled=True,
        evidence_priority_strategy=EvidencePriorityStrategy.LOCAL_FIRST,
    )

    notes = [
        "URL: https://example.com/first-web\nSummary: Performance: medium",
        "--- LOCAL SOURCE 1 ---\nSource: D:/kb/local-priority.md\nUpdatedAt: 2026-03-17T09:00:00+00:00\nContent: Performance: high",
    ]

    appendix, _ = _build_traceable_evidence_appendix(notes, [], cfg)
    local_index = appendix.find("channel=local")
    web_index = appendix.find("channel=web")
    assert_true(local_index != -1 and web_index != -1, "Expected both local and web evidence")
    assert_true(local_index < web_index, "Expected local evidence to be ranked before web in local_first strategy")

    logger.info("PASS: local_first strategy ranks local evidence ahead of web evidence")


def validate_conflict_handling() -> None:
    print_header("PR6 Conflict Handling")

    cfg = Configuration(
        memory_enabled=True,
        evidence_priority_strategy=EvidencePriorityStrategy.LOCAL_FIRST,
    )

    notes = [
        "--- LOCAL SOURCE 1 ---\nSource: D:/kb/device.md\nUpdatedAt: 2026-03-18T08:00:00+00:00\nContent: Battery life: 12h",
        "URL: https://example.com/device-review\nSummary: Battery life: 10h",
    ]

    appendix, _ = _build_traceable_evidence_appendix(notes, [], cfg)
    assert_true("<ConflictHandling>" in appendix, "Expected explicit conflict handling section")
    assert_true("divergent statements" in appendix, "Expected conflict explanation in appendix")

    logger.info("PASS: conflicting evidence is surfaced with uncertainty guidance")


def main() -> int:
    validate_three_channel_fusion_and_citations()
    validate_priority_strategy_local_first()
    validate_conflict_handling()
    logger.info("\nAll PR-6 acceptance checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
