"""Test Turn-3 multimodal probe integration within the ODR LangGraph.

Tests two paths:

  Path 1 — existing report supplied, skip pipeline, run probe only:
    START → multimodal_probe → END

  Path 3 — no report, no probe flag:
    START → clarify_with_user → ... (verifies default behaviour unchanged)

Usage:
    # Path 1 (full run, finds image from existing report)
    uv run python draco_eval/scripts/test_turn3_graph.py --path 1 --task task_003

    # Path 3 (smoke test that default graph still starts normally)
    uv run python draco_eval/scripts/test_turn3_graph.py --path 3 --task task_003
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher, deep_researcher_builder  # noqa: E402

DRACO_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DRACO_DIR / "turn3_outputs"


def load_task(task_id: str) -> dict:
    task_file = DRACO_DIR / "tasks" / f"{task_id}.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    return json.loads(task_file.read_text(encoding="utf-8"))


def load_report(task_id: str) -> str:
    report_file = DRACO_DIR / "reports" / f"{task_id}_v2.md"
    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")
    return report_file.read_text(encoding="utf-8")


async def run_path1(task_id: str) -> None:
    """Path 1: existing report provided → START → multimodal_probe → END."""
    print(f"\n{'='*60}")
    print(f"PATH 1 — existing report, probe only")
    print(f"Task: {task_id}")
    print(f"{'='*60}\n")

    task = load_task(task_id)
    report = load_report(task_id)

    result = await deep_researcher.ainvoke(
        {
            "messages": [HumanMessage(content=task["prompt"])],
            "existing_report": report,
            "task_id": task_id,
        },
        config={
            "configurable": {
                "enable_multimodal_probe": True,
                # probe model (strip provider prefix handled inside the node)
                "multimodal_probe_model": "openai:gpt-4.1",
                # disable clarification so if routing ever falls through it
                # won't hang waiting for user input
                "allow_clarification": False,
            }
        },
    )

    turn3 = result.get("turn3_result")
    print("\nTurn-3 result:")
    print(json.dumps(turn3, indent=2, ensure_ascii=False))

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{task_id}_turn3_graph.json"
    out_path.write_text(json.dumps(turn3, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")


async def run_path3(task_id: str) -> None:
    """Path 3: no existing_report, no probe flag → normal pipeline entry, no probe.

    We only verify the graph routes to clarify_with_user (not multimodal_probe)
    and returns without error. We interrupt immediately after the first node so
    we don't run a full research pipeline.
    """
    print(f"\n{'='*60}")
    print(f"PATH 3 — default behaviour (no probe, no existing report)")
    print(f"Task: {task_id}")
    print(f"{'='*60}\n")

    task = load_task(task_id)

    # Compile fresh with an interrupt_before so we don't run the full pipeline
    graph = deep_researcher_builder.compile(interrupt_before=["write_research_brief"])

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=task["prompt"])]},
        config={
            "configurable": {
                "enable_multimodal_probe": False,
                "allow_clarification": False,
            }
        },
    )

    # Verify probe was NOT triggered
    turn3 = result.get("turn3_result")
    assert turn3 is None, f"Expected no turn3_result in path 3, got: {turn3}"

    # Verify we reached write_research_brief (messages exist, no final_report yet)
    final_report = result.get("final_report", "")
    assert not final_report, "Expected no final_report — pipeline should have been interrupted"

    print("Path 3 OK:")
    print(f"  turn3_result : {turn3}  (expected None ✓)")
    print(f"  final_report : {repr(final_report)[:60]}  (expected empty ✓)")
    print(f"  Interrupted before write_research_brief — default routing confirmed ✓")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Turn-3 graph integration")
    parser.add_argument("--path", required=True, choices=["1", "3"],
                        help="Which path to test: 1=existing report+probe, 3=default no probe")
    parser.add_argument("--task", default="task_003",
                        help="Task ID to use (default: task_003)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.path == "1":
        await run_path1(args.task)
    elif args.path == "3":
        await run_path3(args.task)


if __name__ == "__main__":
    asyncio.run(main())
