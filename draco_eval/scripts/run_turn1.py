"""Run turn-1 of a multi-turn research task WITHOUT checkpointing.

To revert to the checkpointing version, copy run_turn1_with_checkpointing.py
back over this file.

Usage:
    uv run python draco_eval/scripts/run_turn1.py --task draco_eval/tasks/task_002.json
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

from open_deep_research.deep_researcher import deep_researcher_builder  # noqa: E402

DRACO_DIR = Path(__file__).parent.parent
SKIP_NODES = {"LangGraph", ""}


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(task_path: Path):
    task = json.loads(task_path.read_text())
    task_id = task_path.stem
    prompt = task["prompt"]

    output_dir = DRACO_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v1.md"

    print(f"\n{'='*60}")
    print(f"TASK:   {task_id}")
    print(f"OUTPUT: {output_file}")
    print(f"{'='*60}")
    print(f"\nPROMPT:\n{prompt}\n")

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)
    final_result = {}

    graph = deep_researcher_builder.compile()

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config={
            "run_name": f"{task_id}_turn1",
            "configurable": {
                "allow_clarification": False,
            },
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
    raw_notes = final_result.get("notes", [])
    note_count = len(raw_notes.get("value", [])) if isinstance(raw_notes, dict) else len(raw_notes)

    print(f"\n[Debug] Researcher notes collected: {note_count}")

    if report:
        output_file.write_text(report)
        print(f"[Saved] Report written to {output_file}")
    else:
        print("[Error] No report found in agent output — nothing saved.")

    if tracer.latest_run:
        try:
            url = LangSmithClient().get_run_url(run=tracer.latest_run)
            print(f"[LangSmith] {url}")
        except Exception:
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_turn1' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Run turn-1 of an ODR task without checkpointing.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")

    asyncio.run(run(args.task))


if __name__ == "__main__":
    main()
