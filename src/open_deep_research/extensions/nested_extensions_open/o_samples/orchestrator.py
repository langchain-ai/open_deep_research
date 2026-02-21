"""
Orchestrator - Run the research pipeline from the command line (no Chainlit UI).

Usage:
    python orchestrator.py

Useful for debugging the backend without the Chainlit frontend.
Edit the query and settings in main() to test different scenarios.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Ensure src/ is on sys.path so extension imports work
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extensions.agents.master_agent import MasterAgent
from extensions.utils.report_builder import build_html_report


async def run_pipeline(query: str, use_enhanced_research: bool = True) -> None:
    print("Environment loaded and imports ready")
    print(f"Query: {query}")
    print(f"Enhanced research: {use_enhanced_research}")

    # Initialize the master agent
    agent = MasterAgent(
        enable_state_persistence=True,
        storage_type="json",
        use_enhanced_research=use_enhanced_research
    )

    # Run asynchronously
    result = await agent.run_async(query)

    # Extract state and summary
    state = result.get("state", {})
    status = result.get("status", "unknown")
    exec_time = result.get("execution_time", 0.0)
    agents_used = result.get("agents_used", [])
    charts = state.get("charts", [])
    final_report = state.get("final_report", "") or result.get("output", "")
    analysis_output = state.get("analysis_output", "")
    chart_explanations = state.get("chart_explanations", {})
    sources = state.get("sources", [])
    sub_queries = state.get("sub_queries", [])
    conversation_id = state.get("conversation_id", "")

    # # Build HTML report
    # report_path = build_html_report(
    #     display_text=final_report,
    #     analysis_output=analysis_output or "",
    #     figures=charts,
    #     chart_explanations=chart_explanations,
    #     sources=sources,
    #     query=query,
    #     sub_queries=sub_queries,
    #     conversation_id=conversation_id,
    #     src_dir=SRC_DIR,  # ensure paths resolve relative to src/
    # )

    # print(f"HTML report generated: {report_path}")

    # Print summary
    print(f"Status: {status}")
    print(f"Time: {exec_time:.1f}s")
    print(f"Agents used: {agents_used}")
    print(f"Charts: {len(charts)}")
    print(f"Report length: {len(final_report)} chars")


def main():
    # Hand-coded parameters (no argparse)
    query = "perform deep research langchain in usa, china and india. If possible with some visualization."
    use_enhanced_research = True

    try:
        asyncio.run(run_pipeline(query, use_enhanced_research=use_enhanced_research))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")


if __name__ == "__main__":
    main()
