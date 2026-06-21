from __future__ import annotations

import logging

from tests.evaluators import (
    eval_memory_usefulness,
    eval_personalization,
    eval_source_diversity,
)
from tests.run_evaluate import get_pr8_variants, summarize_pr8_regression

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    logger.info(f"\n=== {title} ===")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_new_metric_outputs() -> None:
    print_header("PR8 New Metrics")

    inputs = {
        "messages": [
            {"role": "user", "content": "请用中文简洁回答，并且不要表格。"},
        ]
    }
    outputs = {
        "final_report": "这是一个简洁回答，不使用表格。",
        "confirmed_long_term_preferences": ["language: 中文", "style: 简洁"],
        "api_response": {
            "sources": [
                {"id": "SRC-001", "channel": "memory", "source": "long-term-memory"},
                {"id": "SRC-002", "channel": "local", "source": "D:/kb/one.md"},
                {"id": "SRC-003", "channel": "web", "source": "https://example.com/a"},
            ]
        },
    }

    p = eval_personalization(inputs, outputs)
    m = eval_memory_usefulness(inputs, outputs)
    s = eval_source_diversity(inputs, outputs)

    assert_true(p["key"] == "personalization_score", "personalization key mismatch")
    assert_true(m["key"] == "memory_usefulness_score", "memory usefulness key mismatch")
    assert_true(s["key"] == "source_diversity_score", "source diversity key mismatch")
    assert_true(0.0 <= p["score"] <= 1.0, "personalization score must be normalized")
    assert_true(0.0 <= m["score"] <= 1.0, "memory usefulness score must be normalized")
    assert_true(0.0 <= s["score"] <= 1.0, "source diversity score must be normalized")

    logger.info("PASS: new PR8 metrics produce normalized scores")


def validate_rag_memory_matrix_variants() -> None:
    print_header("PR8 RAG/Memory Matrix")

    variants = get_pr8_variants()
    names = {variant["name"] for variant in variants}

    required = {
        "baseline_off_off",
        "rag_on_memory_off",
        "rag_off_memory_on",
        "rag_on_memory_on",
    }
    assert_true(required.issubset(names), "Missing expected PR8 evaluation variants")

    logger.info("PASS: PR8 matrix includes all required rag/memory combinations")


def validate_regression_summary_baseline() -> None:
    print_header("PR8 Regression Summary")

    rows = [
        {"latency_ms": 100.0, "token_cost": 10.0},
        {"latency_ms": 200.0, "token_cost": 20.0},
        {"latency_ms": 400.0, "token_cost": 30.0},
        {"latency_ms": 800.0, "token_cost": 50.0},
    ]

    summary = summarize_pr8_regression(rows)
    assert_true(summary["count"] == 4, "Expected summary count to match rows")
    assert_true(summary["latency_ms"]["p50"] >= 100.0, "Expected non-zero latency p50")
    assert_true(summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"], "Expected p95 >= p50 for latency")
    assert_true(summary["token_cost"]["p95"] >= summary["token_cost"]["p50"], "Expected p95 >= p50 for token cost")

    logger.info("PASS: regression summary emits P50/P95 baselines")


def main() -> int:
    validate_new_metric_outputs()
    validate_rag_memory_matrix_variants()
    validate_regression_summary_baseline()
    logger.info("\nAll PR-8 acceptance checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
