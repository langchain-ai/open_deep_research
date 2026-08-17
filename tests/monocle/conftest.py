"""Pytest scaffold for the Open Deep Research Monocle test suite.

Enables Monocle tracing, loads the repo `.env`, and exposes
``run_opendeepresearch`` -- the single entry the live tests use to drive the
agent under instrumentation.
"""
import os
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional -- only used to auto-load .env for the live tests
    load_dotenv = None
from monocle_apptrace import setup_monocle_telemetry

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
REPO_ROOT = HERE.parent.parent

# Only export captured spans to the configured exporters (okahu/file) for FAILING
# tests -- a failing trace is the one worth inspecting. This also sidesteps a
# monocle_test_tools export-path detail: on a passing test it re-stamps a status
# attribute onto every captured span, which raises on the *live* tests because
# real (finished) OpenTelemetry spans have immutable attributes. Offline tests
# are unaffected (their spans are loaded dicts). Overridable from the environment.
os.environ.setdefault("MONOCLE_EXPORT_FAILED_TESTS_ONLY", "true")

setup_monocle_telemetry(workflow_name="open-deep-research")

if load_dotenv and (REPO_ROOT / ".env").exists():
    load_dotenv(REPO_ROOT / ".env")


async def run_opendeepresearch(message: str) -> str:
    """Run Open Deep Research once and return its final report text.

    Uses OpenAI-native web search and is cost-capped to a single researcher
    iteration on gpt-4o-mini (override via the ODR_MAX_* env vars).
    """
    from open_deep_research.deep_researcher import deep_researcher

    config = {"configurable": {
        "search_api": "openai",
        "allow_clarification": False,
        "max_researcher_iterations": int(os.environ.get("ODR_MAX_ITERATIONS", 1)),
        "max_concurrent_research_units": int(os.environ.get("ODR_MAX_CONCURRENT_UNITS", 1)),
        "max_react_tool_calls": int(os.environ.get("ODR_MAX_TOOL_CALLS", 1)),
        "research_model": "openai:gpt-4o-mini", "research_model_max_tokens": 4000,
        "summarization_model": "openai:gpt-4o-mini", "summarization_model_max_tokens": 4000,
        "compression_model": "openai:gpt-4o-mini", "compression_model_max_tokens": 4000,
        "final_report_model": "openai:gpt-4o-mini", "final_report_model_max_tokens": 4000,
        "thread_id": f"odr-{uuid.uuid4().hex[:8]}",
    }}
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": message}]}, config=config,
    )
    return result.get("final_report", "")
