"""Smoke-test the event-capture and file-saving logic in run_task.py.

No API calls, no Tavily credits. Feeds fake astream_events into the same
loop that run_task.py uses and verifies the report is written to disk.

Usage:
    uv run python test_save.py draco_eval/tasks/task_002.json --version test
"""

import argparse
import asyncio
import json
from pathlib import Path


FAKE_REPORT = "# Test Report\n\nThis is a fake report to verify file saving works.\n"

SKIP_NODES = {"LangGraph", ""}


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def fake_events():
    """Yield the minimal set of events that run_task.py's loop needs."""
    yield {
        "event": "on_chain_start",
        "name": "clarify_with_user",
        "data": {"input": {"messages": ["test query"]}},
    }
    yield {
        "event": "on_chain_end",
        "name": "clarify_with_user",
        "data": {"output": {"messages": []}},
    }
    yield {
        "event": "on_chain_start",
        "name": "final_report_generation",
        "data": {"input": {}},
    }
    yield {
        "event": "on_chain_end",
        "name": "final_report_generation",
        "data": {"output": {"final_report": FAKE_REPORT, "notes": ["note-1", "note-2"]}},
    }
    yield {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {}},  # intentionally empty — tests primary capture path
    }


async def run(task_path: Path, version: str):
    task_id = task_path.stem

    output_dir = task_path.parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_{version}.md"

    print(f"\n{'='*60}")
    print(f"TASK:    {task_id}")
    print(f"VERSION: {version}")
    print(f"OUTPUT:  {output_file}")
    print(f"{'='*60}")
    print("(using fake events — no API calls)\n")

    final_result = {}

    async for event in fake_events():
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
        print(f"[OK]    File exists: {output_file.exists()}")
        print(f"[OK]    File size:   {output_file.stat().st_size} bytes")
    else:
        print("[Error] No report found — saving logic failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_file", type=Path)
    parser.add_argument("--version", default="test")
    args = parser.parse_args()

    if not args.task_file.exists():
        raise FileNotFoundError(f"Task file not found: {args.task_file}")

    asyncio.run(run(args.task_file, args.version))


if __name__ == "__main__":
    main()
