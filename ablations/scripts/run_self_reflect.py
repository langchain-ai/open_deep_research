"""Run the self-reflection baseline for a single task.

Reads the v1 report and asks the agent to revise it purely through self-reflection
(no external evaluator feedback). Writes the result to
ablations/reports/reports_<slug>_self_reflect/<task_id>_v2.md.

Usage:
    # GPT-4.1 (default)
    uv run python ablations/scripts/run_self_reflect.py \
        --task ablations/tasks/task_002.json

    # GPT-4.1-mini
    uv run python ablations/scripts/run_self_reflect.py \
        --task ablations/tasks/task_002.json \
        --model openai:gpt-4.1-mini
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

SELF_REFLECT_PROMPT = """\
You previously wrote a research report on the following query:

--- ORIGINAL QUERY ---
{original_query}

--- YOUR PREVIOUS REPORT ---
{prev_report}

--- USER FEEDBACK ---
Please reflect on your current report and revise it.
"""


def _model_slug(model: str) -> str:
    """Mirrors run_all_turn1.sh slug logic so folder names always match.

    Examples:
        bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
        anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
        openai:gpt-4.1-mini                                  → gpt41mini
        deepseek:deepseek-v4-flash                           → deepseekv4flash
    """
    import re
    if model.startswith("bedrock:"):
        mid = model[len("bedrock:"):]
        mid = mid.split(":")[0]                          # strip trailing :0
        name = mid.split(".")[-1]                        # strip vendor prefix
        name = re.sub(r"-\d{8}.*", "", name)             # strip date suffix
        return "bedrock_" + name.replace("-", "")
    if model.startswith("anthropic:"):
        name = model[len("anthropic:"):]
        name = re.sub(r"-\d{8}.*", "", name)             # strip date suffix
        return name.replace("-", "")
    # Generic: strip provider prefix and hyphens
    return model.split(":")[-1].replace("-", "")


def _truncate(obj, max_chars=300):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s[:max_chars] + "..." if len(s) > max_chars else s


async def run(task_path: Path, model: str, max_report_tokens: int = 16000):
    task_id = task_path.stem
    slug = _model_slug(model)

    task = json.loads(task_path.read_text())
    original_query = task["prompt"]

    prev_report_path = ABLATIONS_DIR / "reports" / f"reports_{slug}" / f"{task_id}_v1.md"
    if not prev_report_path.exists():
        raise FileNotFoundError(f"v1 report not found at {prev_report_path}")
    prev_report = prev_report_path.read_text().strip()

    combined_message = SELF_REFLECT_PROMPT.format(
        original_query=original_query,
        prev_report=prev_report,
    )

    output_dir = ABLATIONS_DIR / "reports" / f"reports_{slug}_self_reflect"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_id}_v2.md"

    print(f"\n{'='*60}")
    print(f"TASK:   {task_id}")
    print(f"MODE:   self-reflection  (reads v1, writes v2)")
    print(f"MODEL:  {model}")
    print(f"OUTPUT: {output_file}")
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
            "run_name": f"{task_id}_self_reflect",
            "configurable": {
                "allow_clarification": False,
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
            print(f"[LangSmith] Project '{project}' — find run '{task_id}_self_reflect' at smith.langchain.com")


def main():
    parser = argparse.ArgumentParser(description="Self-reflection baseline: revise v1 report without external feedback.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model (e.g. 'openai:gpt-4.1', 'openai:gpt-4.1-mini'). "
             "Controls which reports/reports_<slug>/ folder is read and reports/reports_<slug>_self_reflect/ is written."
    )
    parser.add_argument(
        "--max_report_tokens", type=int, default=16000,
        help="final_report_model_max_tokens (default 16000). Increase if reports are being cut off mid-sentence."
    )
    args = parser.parse_args()

    if not args.task.exists():
        raise FileNotFoundError(f"Task file not found: {args.task}")

    asyncio.run(run(args.task, args.model, args.max_report_tokens))


if __name__ == "__main__":
    main()
