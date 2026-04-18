"""Generate research-gap-aware feedback from a failed evaluation.

Output goes to draco_eval/feedback_<model_slug>/ by default.

Usage:
    # GPT-4.1 (default)
    python draco_eval/scripts/new_generate_feedback.py \
        --evaluation draco_eval/evaluations_gpt4.1/task_002_v1_eval.json \
        --task       draco_eval/tasks/task_002.json

    # GPT-4.1-mini
    python draco_eval/scripts/new_generate_feedback.py \
        --evaluation draco_eval/evaluations_gpt4.1mini/task_002_v1_eval.json \
        --task       draco_eval/tasks/task_002.json \
        --model      openai:gpt-4.1-mini

    # Gemini 2.5 Pro
    python draco_eval/scripts/new_generate_feedback.py \
        --evaluation draco_eval/evaluations_gemini2.5pro/task_002_v1_eval.json \
        --task       draco_eval/tasks/task_002.json \
        --model      google_vertexai:gemini-2.5-pro

    # Dry run (prints prompt without calling API)
    python draco_eval/scripts/new_generate_feedback.py \
        --evaluation draco_eval/evaluations_gpt4.1/task_002_v1_eval.json \
        --task       draco_eval/tasks/task_002.json \
        --dry-run
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

FEEDBACK_MODEL = "gpt-4.1"
DRACO_DIR = Path(__file__).parent.parent
FEEDBACK_PROMPT_PATH = DRACO_DIR / "prompts" / "new_feedback_prompt.txt"


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")


def load_feedback_system_prompt(path: Path) -> str:
    """Extract the k>1 system prompt from new_feedback_prompt.txt."""
    text = path.read_text()
    parts = text.split("=== FEEDBACK PROMPT K>1 ===")
    return parts[1].strip()


def collect_fa_results(results: dict) -> tuple[list[dict], list[dict]]:
    """
    Return FA failures and FA passes separately.
    Passes are included so the model can reason about what the report DID cover.
    Only positive (non-negative) FA criteria are included in passes.
    """
    failures, passes = [], []
    for item in results.get("factual-accuracy", []):
        is_neg = item["is_negative"]
        verdict = item["verdict"]
        failed = (not is_neg and verdict == "UNMET") or (is_neg and verdict == "MET")
        if failed:
            failures.append(item)
        elif not is_neg and verdict == "MET":
            passes.append(item)
    return failures, passes


def collect_bd_results(results: dict) -> tuple[list[dict], list[dict]]:
    """
    Return BD failures and BD passes separately.
    Passes are included so the model can reason about what analytical depth
    the report DID achieve — mirroring the same logic as collect_fa_results().
    Only positive (non-negative) BD criteria are included in passes.
    """
    failures, passes = [], []
    for item in results.get("breadth-and-depth-of-analysis", []):
        is_neg = item["is_negative"]
        verdict = item["verdict"]
        failed = (not is_neg and verdict == "UNMET") or (is_neg and verdict == "MET")
        if failed:
            failures.append(item)
        elif not is_neg and verdict == "MET":
            passes.append(item)
    return failures, passes


def collect_cq_signals(results: dict) -> dict:
    """
    Extract CQ pass/fail as research signals for the inference step.
    These are NOT feedback targets — they help the model infer which
    sources the report found vs missed, and whether it used bad sources.

    Three signal types:
    - found: positive CQ criteria that passed (sources correctly used)
    - missed: positive CQ criteria that failed (required sources not cited)
    - quality_errors: negative CQ criteria that were MET (model cited bad sources
      e.g. Reddit, blogs, social media — a meaningful research process signal)
    """
    found, missed, quality_errors = [], [], []
    for item in results.get("citation-quality", []):
        if item["is_negative"]:
            # Negative CQ MET = model committed a source quality error
            if item["verdict"] == "MET":
                quality_errors.append(item["requirement"])
        else:
            if item["verdict"] == "MET":
                found.append(item["requirement"])
            else:
                missed.append(item["requirement"])
    return {"found": found, "missed": missed, "quality_errors": quality_errors}



def build_user_message(
    original_query: str,
    fa_failures: list[dict],
    fa_passes: list[dict],
    bd_failures: list[dict],
    bd_passes: list[dict],
    cq_signals: dict,
) -> str:
    """
    Build the structured user message for the feedback model.
    Both FA and BD passes are included to support research gap inference.
    FA failures are grouped by topic cluster; FA passes are shown flat.
    CQ signals are framed as research diagnostics, not feedback targets.
    """
    lines = []

    # Query
    lines.append("ORIGINAL QUERY:")
    lines.append(original_query)
    lines.append("")

    # FA passes — ID + requirement (no explanation needed, these passed)
    if fa_passes:
        lines.append("WHAT THE REPORT COVERED CORRECTLY (factual accuracy):")
        for item in fa_passes:
            lines.append(f"  [PASS] [{item['id']}] {item['requirement']}")
        lines.append("")

    # FA failures — ID + requirement + explanation, flat list for model to group itself
    if fa_failures:
        lines.append("WHAT THE REPORT MISSED OR GOT WRONG (factual accuracy):")
        lines.append("(The criterion IDs are included to help you identify related topics — do not treat them as pre-defined groups.)")
        lines.append("")
        for item in fa_failures:
            neg_tag = "  : model committed this error" if item["is_negative"] else ""
            lines.append(f"  - [{item['id']}]{neg_tag}")
            lines.append(f"    Criterion: {item['requirement']}")
            lines.append(f"    Evaluator explanation: {item['explanation']}")
        lines.append("")

    # CQ signals — framed as research diagnostics only
    has_cq = cq_signals["found"] or cq_signals["missed"] or cq_signals["quality_errors"]
    if has_cq:
        lines.append("CITATION / SOURCE SIGNALS (use to infer research process gaps — not feedback targets):")
        if cq_signals["found"]:
            lines.append("  Sources the model found and used correctly:")
            for s in cq_signals["found"]:
                lines.append(f"    [PASS] {s}")
        if cq_signals["missed"]:
            lines.append("  Sources the model missed:")
            for s in cq_signals["missed"]:
                lines.append(f"    [MISS] {s}")
        if cq_signals["quality_errors"]:
            lines.append("  Source quality errors (model cited unreliable sources):")
            for s in cq_signals["quality_errors"]:
                lines.append(f"    [ERROR] {s}")
        lines.append("")

    # BD passes — ID + requirement (no explanation needed, these passed)
    if bd_passes:
        lines.append("WHAT THE REPORT ACHIEVED ANALYTICALLY (breadth-and-depth, correctly handled):")
        for item in bd_passes:
            lines.append(f" [PASS] [{item['id']}] {item['requirement']}")
        lines.append("")

    # BD failures — ID + requirement + explanation, consistent with FA failures
    if bd_failures:
        lines.append("ANALYTICAL DEPTH FAILURES (breadth-and-depth-of-analysis):")
        lines.append("(The criterion IDs are included to help you identify related analytical dimensions — do not treat them as pre-defined groups.)")
        lines.append("")
        for item in bd_failures:
            neg_tag = "  : model committed this error" if item["is_negative"] else ""
            lines.append(f"  - [{item['id']}]{neg_tag}")
            lines.append(f"    Criterion: {item['requirement']}")
            lines.append(f"    Evaluator explanation: {item['explanation']}")
        lines.append("")

    # Final instruction
    lines.append(
        "Using the information above, complete both steps described in your instructions: "
        "first reason about what the pass/fail pattern reveals about the model's research process, "
        "then write the consolidated feedback message."
    )

    return "\n".join(lines)


def extract_feedback_only(response_text: str) -> str:
    """
    Extract just the FEEDBACK section from the model's two-part response.
    Falls back to the full response if the delimiter is not found.
    """
    marker = "FEEDBACK:"
    idx = response_text.find(marker)
    if idx != -1:
        return response_text[idx + len(marker):].strip()
    return response_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate research-gap-aware feedback from a failed evaluation.")
    parser.add_argument("--evaluation", type=Path, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output path for feedback text (optional)")
    parser.add_argument("--output-full", type=Path, default=None, help="Output path for full response including reasoning (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model that generated the evaluated report. "
             "Controls which feedback_<slug>/ folder outputs go to."
    )
    args = parser.parse_args()

    if not args.evaluation.exists():
        raise FileNotFoundError(f"Evaluation not found: {args.evaluation}")
    if not args.task.exists():
        raise FileNotFoundError(f"Task not found: {args.task}")

    # Resolve default output paths
    if args.output is None:
        slug = _model_slug(args.model)
        output_dir = DRACO_DIR / f"feedback_{slug}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{args.evaluation.stem}_feedback.txt"
    if args.output_full is None:
        args.output_full = args.output.parent / f"{args.evaluation.stem}_feedback_full.txt"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_full.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    evaluation = json.loads(args.evaluation.read_text())
    task = json.loads(args.task.read_text())
    original_query = task["prompt"]
    system_prompt = load_feedback_system_prompt(FEEDBACK_PROMPT_PATH)
    results = evaluation["results"]

    # Collect structured inputs
    fa_failures, fa_passes = collect_fa_results(results)
    bd_failures, bd_passes = collect_bd_results(results)
    cq_signals = collect_cq_signals(results)

    total_failures = len(fa_failures) + len(bd_failures)
    if total_failures == 0:
        print("No FA or BD failures found — no feedback needed.")
        return

    print(f"FA failures: {len(fa_failures)} | FA passes (context): {len(fa_passes)}")
    print(f"BD failures: {len(bd_failures)} | BD passes (context): {len(bd_passes)}")
    print(f"CQ signals: {len(cq_signals['found'])} found, {len(cq_signals['missed'])} missed, {len(cq_signals['quality_errors'])} quality errors")
    print()

    user_message = build_user_message(
        original_query, fa_failures, fa_passes, bd_failures, bd_passes, cq_signals
    )

    if args.dry_run:
        print("=== DRY RUN — would send to API ===\n")
        print("SYSTEM:\n", system_prompt)
        print("\nUSER:\n", user_message)
        return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=FEEDBACK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )

    full_response = response.choices[0].message.content.strip()
    feedback_only = extract_feedback_only(full_response)

    # Save both outputs
    args.output.write_text(feedback_only)
    args.output_full.write_text(full_response)

    print("=== RESEARCH GAP ANALYSIS + FEEDBACK ===\n")
    print(full_response)
    print(f"\nFeedback saved to: {args.output}")
    print(f"Full response saved to: {args.output_full}")


if __name__ == "__main__":
    main()