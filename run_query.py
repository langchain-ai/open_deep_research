"""Run a single query through the deep researcher agent and print the final report.

Streams every graph node visit to the terminal and prints the LangSmith trace URL.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client as LangSmithClient

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402 (must load .env first)

QUERY = "What are the key differences between transformer and mamba architecture for LLMs?"

# Nodes that are subgraph entry/exit wrappers — skip to reduce noise
SKIP_NODES = {"LangGraph", ""}


def _truncate(obj, max_chars=300):
    """Truncate a value to max_chars for readable terminal output."""
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def main():
    print(f"\n{'='*60}")
    print(f"QUERY: {QUERY}")
    print(f"{'='*60}\n")

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)

    final_result = {}

    # astream_events yields fine-grained events for every node entry/exit
    async for event in deep_researcher.astream_events(
        {"messages": [HumanMessage(content=QUERY)]},
        config={
            "run_name": "deep-research-test",
            "configurable": {"allow_clarification": False},
            "callbacks": [tracer],
        },
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and name not in SKIP_NODES:
            data_in = event.get("data", {}).get("input")
            print(f"\n>>> NODE START: {name}")
            if data_in:
                print(f"    INPUT: {_truncate(data_in)}")

        elif kind == "on_chain_end" and name not in SKIP_NODES:
            data_out = event.get("data", {}).get("output")
            print(f"<<< NODE END:   {name}")
            if data_out:
                print(f"    OUTPUT: {_truncate(data_out)}")
            if name == "final_report_generation" and isinstance(data_out, dict):
                final_result = data_out

        elif kind == "on_chain_end" and name == "LangGraph":
            # Top-level graph finished — capture the final state (fallback)
            lg_out = event.get("data", {}).get("output", {})
            if isinstance(lg_out, dict) and "final_report" in lg_out:
                final_result = lg_out

    # Print final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60 + "\n")
    report = final_result.get("final_report", "[no report found in output]")
    print(report)

    notes = final_result.get("notes", [])
    print(f"\n[Debug] Researcher notes collected: {len(notes)}")

    # Save report to markdown file
    if report != "[no report found in output]":
        filename = "report.md"
        with open(filename, "w") as f:
            f.write(report)
        print(f"\n[Saved] Report written to {filename}")

    # Print LangSmith trace URL
    if tracer.latest_run:
        try:
            ls_client = LangSmithClient()
            url = ls_client.get_run_url(run=tracer.latest_run)
            print(f"\n[LangSmith] Full trace: {url}")
        except Exception:
            print(f"\n[LangSmith] Project '{project}' — find run 'deep-research-test' at smith.langchain.com")


asyncio.run(main())
