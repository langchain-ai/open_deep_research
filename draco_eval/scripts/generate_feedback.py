"""Generate consolidated user feedback from a failed evaluation.

Usage:
    python draco_eval/scripts/generate_feedback.py \
        --evaluation draco_eval/evaluations/task_002_v1_eval.json \
        --task       draco_eval/tasks/task_002.json

    python draco_eval/scripts/generate_feedback.py \
        --evaluation draco_eval/evaluations/task_002_v1_eval.json \
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

FEEDBACK_MODEL = "gpt-4.1-mini"
DRACO_DIR = Path(__file__).parent.parent
FEEDBACK_PROMPT_PATH = DRACO_DIR / "prompts" / "feedback_prompt.txt"
RUBRIC_CATEGORIES = [
    "factual-accuracy",
    "breadth-and-depth-of-analysis",
    "presentation-quality",
    "citation-quality",
]


def load_feedback_system_prompt(path: Path) -> str:
    """Extract the k>1 system prompt from feedback_prompt.txt."""
    text = path.read_text()
    parts = text.split("=== FEEDBACK PROMPT K>1 ===")
    return parts[1].strip()


def collect_failures(results: dict) -> list[dict]:
    """Return all incorrectly handled criteria across all categories."""
    failures = []
    for category, items in results.items():
        for item in items:
            is_neg = item["is_negative"]
            verdict = item["verdict"]
            # Failed = UNMET positive OR MET negative
            if (not is_neg and verdict == "UNMET") or (is_neg and verdict == "MET"):
                failures.append({**item, "category": category})
    return failures


def build_user_message(original_query: str, failures: list[dict]) -> str:
    """Build the user message to send to the feedback model."""
    lines = [
        f"Original query: {original_query}",
        "",
        "The following issues were found in the report. For each issue, provide "
        "1-2 sentences of natural feedback as if you are the user asking for improvements:",
        "",
    ]

    for i, f in enumerate(failures, 1):
        is_neg = f["is_negative"]
        if is_neg:
            direction = (
                "Frame this as a user request to remove or correct problematic content "
                "that should not be in the report."
            )
        else:
            direction = (
                "Frame this as a user request to add or fix missing information "
                "that should be present in the report."
            )

        lines.append(f"[{i}] Category: {f['category']} | Weight: {f['weight']}")
        lines.append(f"    Evaluator note: {f['explanation']}")
        lines.append(f"    Feedback direction: {direction}")
        lines.append("")

    # lines.append(
    #     "Write one consolidated feedback message covering all the above issues. "
    #     "Be conversational and natural — paraphrase naturally as if a real user is speaking. "
    #     "Do not reproduce the evaluator notes verbatim."
    # )
    lines.append(
    "Write one consolidated feedback message covering all the above issues. "
    "Be conversational and natural — paraphrase naturally as if a real user is speaking. "
    "Do not reproduce the evaluator notes verbatim. "
    "Organize the feedback into 3-4 thematic groups rather than addressing each point "
    "individually. The overall message should feel like one coherent user response, "
    "not a numbered list."      
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate feedback from a failed evaluation.")
    parser.add_argument("--evaluation", type=Path, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output text path (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    args = parser.parse_args()

    if not args.evaluation.exists():
        raise FileNotFoundError(f"Evaluation not found: {args.evaluation}")
    if not args.task.exists():
        raise FileNotFoundError(f"Task not found: {args.task}")

    # Resolve default output path
    if args.output is None:
        output_dir = DRACO_DIR / "feedback"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{args.evaluation.stem}_feedback.txt"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    evaluation = json.loads(args.evaluation.read_text())
    task = json.loads(args.task.read_text())
    original_query = task["prompt"]
    system_prompt = load_feedback_system_prompt(FEEDBACK_PROMPT_PATH)

    failures = collect_failures(evaluation["results"])

    if not failures:
        print("No failures found — all criteria were correctly handled. No feedback needed.")
        return

    print(f"Failed criteria ({len(failures)} total):")
    for f in failures:
        status = "UNMET positive" if not f["is_negative"] else "MET negative"
        print(f"  [{status}] {f['id']}")
    print()

    user_message = build_user_message(original_query, failures)

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

    feedback = response.choices[0].message.content.strip()

    args.output.write_text(feedback)

    print("=== GENERATED FEEDBACK ===\n")
    print(feedback)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
