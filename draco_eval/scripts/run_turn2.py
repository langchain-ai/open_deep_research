"""Run turn-2 of a multi-turn research task by resuming a checkpointed thread.

Injects feedback into the thread created by run_turn1.py (same thread_id = task_id).
LangGraph merges the new message with the saved state and re-runs from START,
so write_research_brief and all downstream nodes see the full conversation history
including the turn-1 report and the new feedback.

Usage:
    uv run python draco_eval/scripts/run_turn2.py \
        --task     draco_eval/tasks/task_002.json \
        --feedback draco_eval/feedback/task_002_v1_eval_feedback.txt
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
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher_builder  # noqa: E402

DRACO_DIR = Path(__file__).parent.parent
CHECKPOINT_DB = DRACO_DIR / "checkpoints" / "checkpoints.db"
SKIP_NODES = {"LangGraph", ""}

FEEDBACK_PREAMBLE = """\
Please revise your previous report based on the following feedback.
Preserve all content and sections not mentioned in the feedback.
Only research and modify sections directly relevant to the feedback provided.

"""


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(task_path: Path, feedback_path: Path):
    task_id = task_path.stem          # must match the thread_id used in turn-1
    thread_id = task_id

    feedback_text = feedback_path.read_text().strip()
    combined_message = FEEDBACK_PREAMBLE + feedback_text

    output_dir = DRACO_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v2.md"

    if not CHECKPOINT_DB.exists():
        raise FileNotFoundError(
            f"Checkpoint DB not found at {CHECKPOINT_DB}. "
            "Run run_turn1.py first to create the checkpoint."
        )

    print(f"\n{'='*60}")
    print(f"TASK:      {task_id}")
    print(f"THREAD ID: {thread_id}  (resuming turn-1 checkpoint)")
    print(f"FEEDBACK:  {feedback_path}")
    print(f"OUTPUT:    {output_file}")
    print(f"{'='*60}")
    print(f"\nFEEDBACK MESSAGE:\n{combined_message}\n")

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)
    final_result = {}

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = deep_researcher_builder.compile(checkpointer=checkpointer)

        # Pass only the new message — LangGraph merges it with the saved thread state
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=combined_message)]},
            config={
                "run_name": f"{task_id}_turn2",
                "configurable": {
                    "thread_id": thread_id,
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
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_turn2' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Run turn-2 of an ODR task by resuming a checkpoint.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--feedback", type=Path, required=True, help="Path to feedback text file")
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")
    if not args.feedback.exists():
        raise FileNotFoundError(f"Feedback file not found: {args.feedback}")

    asyncio.run(run(args.task, args.feedback))


if __name__ == "__main__":
    main()
