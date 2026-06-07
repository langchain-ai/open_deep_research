"""Run turn-N (2 or 3) of a multi-turn research task.

Reads the v{turn-1} report from ablations/reports_<slug>/ and writes the v{turn}
report to the same slug-namespaced folder.

Usage:
    # Turn-2, GPT-4.1 (default)
    uv run python ablations/scripts/run_turn2.py \
        --task     ablations/tasks/task_002.json \
        --feedback ablations/feedback_gpt4.1/task_002_v1_eval_feedback.txt

    # Turn-3, GPT-4.1 (reads v2 report + v2 feedback, writes v3)
    uv run python ablations/scripts/run_turn2.py \
        --task     ablations/tasks/task_002.json \
        --feedback ablations/feedback_gpt4.1/task_002_v2_eval_feedback.txt \
        --turn     3

    # GPT-4.1-mini
    uv run python ablations/scripts/run_turn2.py \
        --task     ablations/tasks/task_002.json \
        --feedback ablations/feedback_gpt4.1mini/task_002_v1_eval_feedback.txt \
        --model    openai:gpt-4.1-mini

    # Gemini 2.5 Pro via Vertex AI
    uv run python ablations/scripts/run_turn2.py \
        --task     ablations/tasks/task_002.json \
        --feedback ablations/feedback_gemini2.5pro/task_002_v1_eval_feedback.txt \
        --model    google_vertexai:gemini-2.5-pro
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

TURN_PROMPT = """\
You previously wrote a research report on the following query:

--- ORIGINAL QUERY ---
{original_query}

--- YOUR PREVIOUS REPORT ---
{prev_report}

--- USER FEEDBACK ---
{feedback}

Please revise your report based on the feedback above. The feedback identifies gaps \
and weaknesses in your previous report — areas where your research was incomplete, \
incorrect, or lacking depth. Your revision should address these gaps while retaining \
everything else from your previous report that remains valid.
"""


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(task_path: Path, feedback_path: Path, model: str, turn: int, max_report_tokens: int = 16000):
    task_id = task_path.stem
    slug = _model_slug(model)
    prev_turn = turn - 1

    task = json.loads(task_path.read_text())
    original_query = task["prompt"]

    prev_report_path = DRACO_DIR / f"reports_{slug}" / f"{task_id}_v{prev_turn}.md"
    if not prev_report_path.exists():
        raise FileNotFoundError(f"v{prev_turn} report not found at {prev_report_path}")
    prev_report = prev_report_path.read_text().strip()

    feedback_text = feedback_path.read_text().strip()

    combined_message = TURN_PROMPT.format(
        original_query=original_query,
        prev_report=prev_report,
        feedback=feedback_text,
    )

    output_dir = DRACO_DIR / f"reports_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v{turn}.md"

    print(f"\n{'='*60}")
    print(f"TASK:     {task_id}")
    print(f"TURN:     {turn}  (reads v{prev_turn}, writes v{turn})")
    print(f"MODEL:    {model}")
    print(f"FEEDBACK: {feedback_path}")
    print(f"OUTPUT:   {output_file}")
    print(f"{'='*60}")

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)
    final_result = {}

    graph = deep_researcher_builder.compile()

    async for event in graph.astream_events(
        {
            "messages": [HumanMessage(content=combined_message)],
            "task_id": task_id,
        },
        config={
            "run_name": f"{task_id}_turn{turn}",
            "configurable": {
                "allow_clarification": False,
                "enable_multimodal_probe": False,
                "multimodal_probe_model": "openai:gpt-4.1",
                "research_model": model,
                "compression_model": model,
                "final_report_model": model,
                "final_report_model_max_tokens": max_report_tokens,
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
            elif name == "multimodal_probe" and isinstance(data_out, dict):
                # Capture turn3_result directly from probe node output
                if "turn3_result" in data_out:
                    final_result["turn3_result"] = data_out["turn3_result"]

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

    # Save multimodal probe result if available
    turn3_result = final_result.get("turn3_result")
    if turn3_result:
        # Normalise to dict so we can update report_path
        if hasattr(turn3_result, "model_dump"):
            turn3_result = turn3_result.model_dump()
        elif not isinstance(turn3_result, dict):
            turn3_result = dict(turn3_result)

        # Override report_path to point to the actual saved file
        turn3_result["report_path"] = str(output_file.resolve())

        probe_dir = DRACO_DIR / f"multimodal_probe_outputs_{slug}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        images_dir = probe_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe_dir / f"{task_id}_probe.json"
        probe_path.write_text(json.dumps(turn3_result, indent=2, ensure_ascii=False))
        print(f"[Saved] Multimodal probe result written to {probe_path}")
        img = turn3_result.get("local_image_path", "")
        if img:
            print(f"[Saved] Image at {img}")
    else:
        print("[Probe] No multimodal probe result found in output.")

    if tracer.latest_run:
        try:
            url = LangSmithClient().get_run_url(run=tracer.latest_run)
            print(f"[LangSmith] {url}")
        except Exception:
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_turn{turn}' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Run turn-N (2 or 3) of an ODR task without checkpointing.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--feedback", type=Path, required=True, help="Path to feedback text file")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model (e.g. 'openai:gpt-4.1', 'openai:gpt-4.1-mini', 'google_vertexai:gemini-2.5-pro'). "
             "Controls which reports_<slug>/ folder is used."
    )
    parser.add_argument(
        "--turn", type=int, default=2, choices=[2, 3],
        help="Which turn to run (2 or 3). Turn N reads v{N-1} report and writes v{N} report (default: 2)."
    )
    parser.add_argument(
        "--max_report_tokens", type=int, default=16000,
        help="final_report_model_max_tokens (default 16000). Increase if reports are being cut off mid-sentence."
    )
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")
    if not args.feedback.exists():
        raise FileNotFoundError(f"Feedback file not found: {args.feedback}")

    asyncio.run(run(args.task, args.feedback, args.model, args.turn, args.max_report_tokens))


if __name__ == "__main__":
    main()