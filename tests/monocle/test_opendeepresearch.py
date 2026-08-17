"""Trace-based behavioural tests for Open Deep Research, using Monocle Test Tools.

Each test asserts against the Monocle trace a run emits -- which agent ran, what
it was asked, what it produced, and its token/duration cost. Four offline tests
replay recorded good traces (fast, no keys), one per curated question; a single
live test runs the agent end-to-end.

    pytest tests/monocle/ -k "not live"   # offline, no keys
    pytest tests/monocle/                 # includes the live run (needs OPENAI_API_KEY)

Open Deep Research runs its search inside the model call (OpenAI-native web
search in the `openai.resources.responses` model-api spans), so its traces carry
NO `agentic.tool.invocation` spans. The tests therefore assert the agent
invocation, output, and budgets that actually exist in the trace, and do not
assert tool calls.
"""
import asyncio
import os

import pytest
from monocle_test_tools import TraceAssertion

from conftest import TRACES, run_opendeepresearch

# Recorded good traces (captured from this repo under monocle_apptrace 0.8.8),
# one per curated question.
TRACE_SEASONS = str(TRACES / "monocle_trace_open-deep-research_44ab7d2b1a08a7a1de1413be4b08dc46_2026-07-09_12.16.43.json")
TRACE_ENERGY = str(TRACES / "monocle_trace_open-deep-research_cebe8a23280881e45b22640af87f6e00_2026-07-09_12.16.58.json")
TRACE_TIDES = str(TRACES / "monocle_trace_open-deep-research_2198c840ff4f156e64ea911eef0a5c71_2026-07-09_12.17.19.json")
TRACE_TCP_UDP = str(TRACES / "monocle_trace_open-deep-research_5d853757970d9be9f527ce95d1b1e4dc_2026-07-09_12.17.44.json")


# --- Offline: replay recorded good traces, one per curated question -------

def test_earth_seasons(monocle_trace_asserter: TraceAssertion):
    """What causes Earth's seasons (explainer). Real trace: 3,427 total tokens,
    ~14.6s workflow duration; agent = LangGraph (CompiledStateGraph)."""
    monocle_trace_asserter.with_trace_source("file", trace_path=TRACE_SEASONS)

    monocle_trace_asserter.called_agent("LangGraph").contains_output("Earth's Seasons")
    monocle_trace_asserter.contains_any_output("season", "seasons", "tilt", "axial", "Earth")
    monocle_trace_asserter.under_token_limit(20_000)
    monocle_trace_asserter.under_duration(60, span_type="workflow")

    # Eval layer (deferred -- set OKAHU_API_KEY and uncomment to enable):
    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")


def test_renewable_vs_nonrenewable(monocle_trace_asserter: TraceAssertion):
    """Renewable vs. nonrenewable energy sources (comparison). Real trace: 5,054
    total tokens, ~16.9s workflow duration; agent = LangGraph."""
    monocle_trace_asserter.with_trace_source("file", trace_path=TRACE_ENERGY)

    monocle_trace_asserter.called_agent("LangGraph").contains_output("Renewable and Nonrenewable Energy Sources")
    monocle_trace_asserter.contains_any_output("renewable", "nonrenewable", "energy")
    monocle_trace_asserter.under_token_limit(20_000)
    monocle_trace_asserter.under_duration(60, span_type="workflow")

    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")


def test_ocean_tides(monocle_trace_asserter: TraceAssertion):
    """What causes ocean tides (explainer). Real trace: 5,445 total tokens,
    ~21.4s workflow duration; agent = LangGraph."""
    monocle_trace_asserter.with_trace_source("file", trace_path=TRACE_TIDES)

    monocle_trace_asserter.called_agent("LangGraph").contains_output("Ocean Tides")
    monocle_trace_asserter.contains_any_output("tide", "tides", "moon", "gravitational")
    monocle_trace_asserter.under_token_limit(20_000)
    monocle_trace_asserter.under_duration(60, span_type="workflow")

    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")


def test_tcp_vs_udp(monocle_trace_asserter: TraceAssertion):
    """TCP vs. UDP (comparison). Real trace: 5,041 total tokens, ~21.3s workflow
    duration; agent = LangGraph."""
    monocle_trace_asserter.with_trace_source("file", trace_path=TRACE_TCP_UDP)

    monocle_trace_asserter.called_agent("LangGraph").contains_output("TCP and UDP")
    monocle_trace_asserter.contains_any_output("TCP", "UDP", "protocol", "packet")
    monocle_trace_asserter.under_token_limit(20_000)
    monocle_trace_asserter.under_duration(60, span_type="workflow")

    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")


# --- Live: run the agent end-to-end ---------------------------------------
# Output text varies run to run, so this asserts structure + budget, with
# contains_any_output kept phrasing-robust. Uses OpenAI-native search (only
# OPENAI_API_KEY needed); the runner is cost-capped to one researcher iteration.

def test_tcp_vs_udp_live(monocle_trace_asserter: TraceAssertion):
    """Comparison path, run live: the main differences between TCP and UDP."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set -- cannot run the live open-deep-research graph")

    asyncio.run(monocle_trace_asserter.validator.test_workflow_async(
        run_opendeepresearch,
        {"test_input": ("What are the main differences between the TCP and UDP protocols?",)},
    ))

    monocle_trace_asserter.called_agent("LangGraph")
    monocle_trace_asserter.contains_any_output("TCP", "UDP", "protocol", "packet")
    monocle_trace_asserter.under_token_limit(500_000)
    monocle_trace_asserter.under_duration(300, units="seconds", span_type="workflow")

    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")
