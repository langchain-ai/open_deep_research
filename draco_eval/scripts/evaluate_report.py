"""Evaluate a research report against a task's rubric criteria.

Usage:
    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports/task_002_v1.md \
        --task   draco_eval/tasks/task_002.json

    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports/task_002_v1.md \
        --task   draco_eval/tasks/task_002.json \
        --output draco_eval/evaluations/task_002_v1_eval.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "gpt-4.1"
DRACO_DIR = Path(__file__).parent.parent
JUDGE_PROMPT_PATH = DRACO_DIR / "prompts" / "judge_prompt.txt"
RUBRIC_CATEGORIES = [
    "factual-accuracy",
    "breadth-and-depth-of-analysis",
    "presentation-quality",
    "citation-quality",
]


def load_judge_prompt(path: Path) -> tuple[str, str]:
    """Split judge_prompt.txt into (system_prompt, user_prompt_template)."""
    text = path.read_text()
    # Split on markers, strip surrounding whitespace from each part
    parts = text.split("=== USER PROMPT ===")
    user_template = parts[1].strip()
    system_part = parts[0].split("=== SYSTEM PROMPT ===")[1].strip()
    return system_part, user_template


def evaluate_criterion(
    client: OpenAI,
    system_prompt: str,
    user_template: str,
    criterion_type: str,
    criterion_requirement: str,
    original_query: str,
    report_text: str,
) -> dict:
    """Call OpenAI to evaluate a single criterion. Returns {verdict, explanation}."""
    # Use explicit replacement instead of str.format() to avoid KeyError
    # if the report text contains curly braces (e.g. JSON examples, tables).
    user_msg = user_template
    user_msg = user_msg.replace("{criterion_type}", criterion_type)
    user_msg = user_msg.replace("{criterion_requirement}", criterion_requirement)
    user_msg = user_msg.replace("{original_query}", original_query)
    user_msg = user_msg.replace("{report_text}", report_text)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    return {
        "verdict": parsed["criterion_status"],
        "explanation": parsed["explanation"],
    }


def compute_summary(results: dict) -> dict:
    total = 0
    positive = 0
    negative = 0
    correctly_handled = 0
    numerator = 0
    denominator = 0

    for category_items in results.values():
        for item in category_items:
            total += 1
            is_neg = item["is_negative"]
            weight = item["weight"]
            verdict = item["verdict"]

            if is_neg:
                negative += 1
            else:
                positive += 1
                denominator += weight  # only positive weights go in denominator

            # Correctly handled: MET for positive, UNMET for negative
            correct = (not is_neg and verdict == "MET") or (is_neg and verdict == "UNMET")
            if correct:
                correctly_handled += 1
                numerator += abs(weight)

    normalized_score = round((numerator / denominator) * 100, 2) if denominator else 0.0

    return {
        "total_criteria": total,
        "positive_criteria": positive,
        "negative_criteria": negative,
        "correctly_handled": correctly_handled,
        "normalized_score": normalized_score,
        "simple_pass_count": correctly_handled,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a research report against rubric criteria.")
    parser.add_argument("--report", type=Path, required=True, help="Path to report markdown file")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"Report not found: {args.report}")
    if not args.task.exists():
        raise FileNotFoundError(f"Task not found: {args.task}")

    # Resolve default output path
    if args.output is None:
        output_dir = DRACO_DIR / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{args.report.stem}_eval.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    task = json.loads(args.task.read_text())
    report_text = args.report.read_text()
    system_prompt, user_template = load_judge_prompt(JUDGE_PROMPT_PATH)
    original_query = task["prompt"]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = {cat: [] for cat in RUBRIC_CATEGORIES}

    # Evaluate every criterion across all categories
    for category in RUBRIC_CATEGORIES:
        criteria = task.get("rubrics", {}).get(category, [])
        for criterion in criteria:
            cid = criterion["id"]
            requirement = criterion["requirement"]
            weight = criterion["weight"]
            is_negative = weight < 0
            criterion_type = "negative" if is_negative else "positive"

            print(f"  Evaluating [{category}] {cid} ...", end=" ", flush=True)

            outcome = evaluate_criterion(
                client=client,
                system_prompt=system_prompt,
                user_template=user_template,
                criterion_type=criterion_type,
                criterion_requirement=requirement,
                original_query=original_query,
                report_text=report_text,
            )

            verdict = outcome["verdict"]
            print(verdict)

            results[category].append({
                "id": cid,
                "requirement": requirement,
                "weight": weight,
                "is_negative": is_negative,
                "verdict": verdict,
                "explanation": outcome["explanation"],
            })

    summary = compute_summary(results)

    output = {
        "task_id": task["task_id"],
        "domain": task.get("domain", ""),
        "report_file": args.report.name,
        "summary": summary,
        "results": results,
    }

    args.output.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print(f"Total criteria:     {summary['total_criteria']}")
    print(f"Correctly handled:  {summary['correctly_handled']}")
    print(f"Normalized score:   {summary['normalized_score']}%")
    print(f"Saved to:           {args.output}")


if __name__ == "__main__":
    main()
