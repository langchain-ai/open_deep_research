"""Run turn-1 of a multi-turn research task WITHOUT checkpointing.

Output goes to ablations/reports_<model_slug>[_<save_name>]/<task_id>_v1.md.

Usage:
    # GPT-4.1 (default)
    uv run python ablations/scripts/run_turn1.py --task ablations/tasks/task_002.json

    # GPT-4.1-mini
    uv run python ablations/scripts/run_turn1.py \
        --task  ablations/tasks/task_002.json \
        --model openai:gpt-4.1-mini

    # Gemini 2.5 Pro via Vertex AI
    uv run python ablations/scripts/run_turn1.py \
        --task  ablations/tasks/task_002.json \
        --model google_vertexai:gemini-2.5-pro

    # Claude Sonnet 4.5 via AWS Bedrock (requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
    uv run python ablations/scripts/run_turn1.py \
        --task  ablations/tasks/task_002.json \
        --model "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # With save_name suffix (outputs go to reports_gpt4.1_ablation_1/)
    uv run python ablations/scripts/run_turn1.py \
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

DRACO_DIR = Path(__file__).parent.parent
SKIP_NODES = {"LangGraph", ""}


def _model_slug(model: str) -> str:
    """Convert a model string to a filesystem-safe slug.

    openai:gpt-4.1                                       → gpt4.1
    google_vertexai:gemini-2.5-pro                       → gemini2.5pro
    anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
    bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
    """
    if model.startswith("bedrock:"):
        # e.g. bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
        # parts[1] = 'us.anthropic.claude-sonnet-4-5-20250929-v1'
        model_id = model.split(":")[1].split(".")[-1]   # 'claude-sonnet-4-5-20250929-v1'
        # drop the date+version suffix (first all-digit segment of length 8+)
        segments = model_id.split("-")
        name = "-".join(s for s in segments
                        if not (s.isdigit() and len(s) >= 8)
                        and not (s.startswith("v") and s[1:].isdigit()))
        return "bedrock_" + name.replace("-", "")       # 'bedrock_claudesonnet45'
    if model.startswith("anthropic:"):
        name = model[len("anthropic:"):]                # 'claude-haiku-4-5-20251001'
        segments = name.split("-")
        name = "-".join(s for s in segments if not (s.isdigit() and len(s) >= 8))
        return name.replace("-", "")                    # 'claudehaiku45'
    return model.split(":")[-1].replace("-", "")


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(task_path: Path, model: str, save_name: str, concurrency: int, iterations: int, max_report_tokens: int = 16000):
    task = json.loads(task_path.read_text())
    task_id = task_path.stem
    prompt = task["prompt"]
    slug = _model_slug(model)
    suffix = f"_{save_name}" if save_name else ""

    output_dir = DRACO_DIR / f"reports_{slug}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v1.md"

    print(f"\n{'='*60}")
    print(f"TASK:   {task_id}")
    print(f"MODEL:  {model}")
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
                "research_model": model,
                "compression_model": model,
                "final_report_model": model,
                "max_concurrent_research_units": concurrency,
                "max_researcher_iterations": iterations,
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
    parser = argparse.ArgumentParser(description="Run turn-1 of an ablation task without checkpointing.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model (e.g. 'openai:gpt-4.1', 'openai:gpt-4.1-mini', 'google_vertexai:gemini-2.5-pro'). "
             "Controls which reports_<slug>/ folder outputs go to."
    )
    parser.add_argument(
        "--save_name", default="",
        help="Optional suffix for output folders (e.g. 'ablation_1' → reports_gpt4.1_ablation_1/). "
             "Useful to separate different ablation runs."
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="max_concurrent_research_units (default 5). Reduce to 1-2 if hitting rate limits."
    )
    parser.add_argument(
        "--iterations", type=int, default=6,
        help="max_researcher_iterations (default 6). Reduce to 3-4 to cut token usage."
    )
    parser.add_argument(
        "--max_report_tokens", type=int, default=16000,
        help="final_report_model_max_tokens (default 16000). Increase if reports are being cut off mid-sentence."
    )
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")

    asyncio.run(run(args.task, args.model, args.save_name, args.concurrency, args.iterations, args.max_report_tokens))


if __name__ == "__main__":
    main()
