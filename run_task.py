"""Run a deep research task from a JSON task file and save the report as markdown.

Usage:
    uv run python run_task.py draco_eval/tasks/task_002.json
    uv run python run_task.py draco_eval/tasks/task_002.json --version v2

The report is saved to draco_eval/reports/<task_id>_<version>.md
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client as LangSmithClient

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402


def load_task(task_path: Path) -> dict:
    with open(task_path) as f:
        return json.load(f)


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


SKIP_NODES = {"LangGraph", ""}


async def run(task_path: Path, version: str):
    task = load_task(task_path)
    task_id = task_path.stem          # e.g. "task_002"
    prompt = task["prompt"]

    output_dir = task_path.parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_{version}.md"

    print(f"\n{'='*60}")
    print(f"TASK:    {task_id}")
    print(f"VERSION: {version}")
    print(f"OUTPUT:  {output_file}")
    print(f"{'='*60}")
    print(f"\nPROMPT:\n{prompt}\n")

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)
    final_result = {}

    async for event in deep_researcher.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config={
            "run_name": f"{task_id}_{version}",
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
            lg_out = event.get("data", {}).get("output", {})
            if isinstance(lg_out, dict) and "final_report" in lg_out:
                final_result = lg_out

    report = final_result.get("final_report", "")
    notes = final_result.get("notes", [])

    print(f"\n[Debug] Researcher notes collected: {len(notes)}")

    if report:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"[Saved] Report written to {output_file}")
    else:
        print("[Error] No report found in agent output — nothing saved.")

    # LangSmith trace URL
    if tracer.latest_run:
        try:
            url = LangSmithClient().get_run_url(run=tracer.latest_run)
            print(f"[LangSmith] Full trace: {url}")
        except Exception:
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_{version}' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Run a deep research task from a JSON file.")
    parser.add_argument("task_file", type=Path, help="Path to the task JSON file")
    parser.add_argument("--version", default="v1", help="Report version suffix (default: v1)")
    args = parser.parse_args()

    if not args.task_file.exists():
        raise FileNotFoundError(f"Task file not found: {args.task_file}")

    asyncio.run(run(args.task_file, args.version))


if __name__ == "__main__":
    main()
