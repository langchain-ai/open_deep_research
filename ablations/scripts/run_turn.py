"""Run any turn (1, 2, or 3) of a multi-turn research task.

Turn 1: runs the agent on the raw task prompt.
Turn 2/3: runs the agent with the previous report and evaluator feedback injected
          as context, asking it to revise the report.

Output goes to ablations/reports/reports_<model_slug>[_<save_name>]/<task_id>_v{turn}.md.

Usage:
    # Turn-1, GPT-4.1 (default)
    uv run python ablations/scripts/run_turn.py \
        --task ablations/tasks/task_002.json

    # Turn-2, GPT-4.1 (reads v1 report + v1 feedback, writes v2)
    uv run python ablations/scripts/run_turn.py \
        --task     ablations/tasks/task_002.json \
        --turn     2 \
        --feedback ablations/feedback/feedback_gpt4.1/task_002_v1_eval_feedback.txt

    # Turn-3, GPT-4.1 (reads v2 report + v2 feedback, writes v3)
    uv run python ablations/scripts/run_turn.py \
        --task     ablations/tasks/task_002.json \
        --turn     3 \
        --feedback ablations/feedback/feedback_gpt4.1/task_002_v2_eval_feedback.txt

    # GPT-4.1-mini, turn-1
    uv run python ablations/scripts/run_turn.py \
        --task  ablations/tasks/task_002.json \
        --model openai:gpt-4.1-mini

    # Gemini 2.5 Pro via Vertex AI
    uv run python ablations/scripts/run_turn.py \
        --task  ablations/tasks/task_002.json \
        --model google_vertexai:gemini-2.5-pro

    # Claude Sonnet 4.5 via AWS Bedrock
    uv run python ablations/scripts/run_turn.py \
        --task  ablations/tasks/task_002.json \
        --model "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # With save_name suffix (outputs go to reports/reports_gpt4.1_ablation_1/)
    uv run python ablations/scripts/run_turn.py \
        --task      ablations/tasks/task_002.json \
        --save_name ablation_1
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

ABLATIONS_DIR = Path(__file__).parent.parent
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
    """Convert a model string to a filesystem-safe slug.

    openai:gpt-4.1                                       → gpt4.1
    google_vertexai:gemini-2.5-pro                       → gemini2.5pro
    anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
    bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
    """
    if model.startswith("bedrock:"):
        model_id = model.split(":")[1].split(".")[-1]
        segments = model_id.split("-")
        name = "-".join(s for s in segments
                        if not (s.isdigit() and len(s) >= 8)
                        and not (s.startswith("v") and s[1:].isdigit()))
        return "bedrock_" + name.replace("-", "")
    if model.startswith("anthropic:"):
        name = model[len("anthropic:"):]
        segments = name.split("-")
        name = "-".join(s for s in segments if not (s.isdigit() and len(s) >= 8))
        return name.replace("-", "")
    return model.split(":")[-1].replace("-", "")


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(
    task_path: Path,
    turn: int,
    feedback_path: Path | None,
    model: str,
    save_name: str,
    concurrency: int,
    iterations: int,
    max_report_tokens: int,
):
    task = json.loads(task_path.read_text())
    task_id = task_path.stem
    slug = _model_slug(model)
    suffix = f"_{save_name}" if save_name else ""

    output_dir = ABLATIONS_DIR / "reports" / f"reports_{slug}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v{turn}.md"

    print(f"\n{'='*60}")
    print(f"TASK:   {task_id}")
    print(f"TURN:   {turn}")
    print(f"MODEL:  {model}")
    print(f"OUTPUT: {output_file}")
    print(f"{'='*60}")

    # Build input message and graph state
    if turn == 1:
        message_content = task["prompt"]
        print(f"\nPROMPT:\n{message_content}\n")
        graph_input = {"messages": [HumanMessage(content=message_content)]}
    else:
        prev_report_path = output_dir / f"{task_id}_v{turn - 1}.md"
        if not prev_report_path.exists():
            raise FileNotFoundError(f"v{turn - 1} report not found: {prev_report_path}")
        prev_report = prev_report_path.read_text().strip()
        feedback_text = feedback_path.read_text().strip()  # type: ignore[union-attr]
        message_content = TURN_PROMPT.format(
            original_query=task["prompt"],
            prev_report=prev_report,
            feedback=feedback_text,
        )
        print(f"FEEDBACK: {feedback_path}\n")
        graph_input = {"messages": [HumanMessage(content=message_content)], "task_id": task_id}

    # Build configurable — turn-1 exposes concurrency/iteration controls
    configurable: dict = {
        "allow_clarification": False,
        "research_model": model,
        "compression_model": model,
        "final_report_model": model,
        "final_report_model_max_tokens": max_report_tokens,
    }
    if turn == 1:
        configurable["max_concurrent_research_units"] = concurrency
        configurable["max_researcher_iterations"] = iterations

    project = os.getenv("LANGSMITH_PROJECT", "default")
    tracer = LangChainTracer(project_name=project)
    final_result: dict = {}

    graph = deep_researcher_builder.compile()

    async for event in graph.astream_events(
        graph_input,
        config={
            "run_name": f"{task_id}_turn{turn}",
            "configurable": configurable,
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
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_turn{turn}' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Run any turn (1, 2, or 3) of an ablation task.")
    parser.add_argument("--task", type=Path, required=True,
                        help="Path to task JSON file")
    parser.add_argument("--turn", type=int, default=1, choices=[1, 2, 3],
                        help="Which turn to run (default: 1). Turn N reads v{N-1} report + feedback and writes v{N}.")
    parser.add_argument("--feedback", type=Path, default=None,
                        help="Path to evaluator feedback text file. Required for turns 2 and 3.")
    parser.add_argument("--model", default="openai:gpt-4.1",
                        help="Agent model (e.g. 'openai:gpt-4.1', 'openai:gpt-4.1-mini', "
                             "'google_vertexai:gemini-2.5-pro'). Controls which reports_<slug>/ folder is used.")
    parser.add_argument("--save_name", default="",
                        help="Optional suffix for output folders (e.g. 'ablation_1' → reports_gpt4.1_ablation_1/).")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="max_concurrent_research_units for turn-1 (default 5). Reduce if hitting rate limits.")
    parser.add_argument("--iterations", type=int, default=6,
                        help="max_researcher_iterations for turn-1 (default 6). Reduce to cut token usage.")
    parser.add_argument("--max_report_tokens", type=int, default=16000,
                        help="final_report_model_max_tokens (default 16000). Increase if reports are cut off.")
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")
    if args.turn > 1 and args.feedback is None:
        parser.error(f"--feedback is required for turn {args.turn}")
    if args.feedback is not None and not args.feedback.exists():
        raise FileNotFoundError(f"Feedback file not found: {args.feedback}")

    asyncio.run(run(
        args.task, args.turn, args.feedback, args.model,
        args.save_name, args.concurrency, args.iterations, args.max_report_tokens,
    ))


if __name__ == "__main__":
    main()