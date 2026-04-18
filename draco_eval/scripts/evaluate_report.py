"""Evaluate a research report against a task's rubric criteria.

Output goes to draco_eval/evaluations_<model_slug>/ by default.

Usage:
    # GPT-4.1 (default)
    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports_gpt4.1/task_002_v1.md \
        --task   draco_eval/tasks/task_002.json

    # GPT-4.1-mini
    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports_gpt4.1mini/task_002_v1.md \
        --task   draco_eval/tasks/task_002.json \
        --model  openai:gpt-4.1-mini

    # Gemini 2.5 Pro
    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports_gemini2.5pro/task_002_v1.md \
        --task   draco_eval/tasks/task_002.json \
        --model  google_vertexai:gemini-2.5-pro

    # Override output path explicitly
    python draco_eval/scripts/evaluate_report.py \
        --report draco_eval/reports_gpt4.1/task_039_v2.md \
        --task   draco_eval/tasks/task_039.json \
        --output draco_eval/evaluations_gpt4.1/task_039_v2_eval.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "gpt-5.2-2025-12-11"
DRACO_DIR = Path(__file__).parent.parent
JUDGE_PROMPT_PATH = DRACO_DIR / "prompts" / "judge_prompt.txt"


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")
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
        reasoning_effort="none",
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    verdict = (
        parsed.get("criterion_status")
        or parsed.get("status")
        or parsed.get("verdict")
    )
    if verdict not in ("MET", "UNMET"):
        raise ValueError(f"Unexpected verdict '{verdict}' from model. Raw: {raw}")
    return {
        "verdict": verdict,
        "explanation": parsed["explanation"],
    }


def compute_normalized_score(items: list) -> float:
    """Compute normalized score per the paper's formula:
        raw = sum of w_i for all criteria where verdict == MET
        denominator = sum of max(0, w_i) across all criteria
        normalized = max(0, min(1, raw / denominator)) * 100
    """
    raw = sum(item["weight"] for item in items if item["verdict"] == "MET")
    denominator = sum(max(0, item["weight"]) for item in items)
    if denominator == 0:
        return 0.0
    return round(max(0.0, min(1.0, raw / denominator)) * 100, 2)


def compute_category_summary(items: list) -> dict:
    """Compute pass/total and normalised score for a single rubric category."""
    correctly_handled = sum(
        1 for item in items
        if (not item["is_negative"] and item["verdict"] == "MET")
        or (item["is_negative"] and item["verdict"] == "UNMET")
    )
    return {
        "total": len(items),
        "correctly_handled": correctly_handled,
        "normalized_score": compute_normalized_score(items),
    }


def compute_summary(results: dict) -> dict:
    total = 0
    positive = 0
    negative = 0
    correctly_handled = 0
    all_items = []

    for category_items in results.values():
        for item in category_items:
            total += 1
            all_items.append(item)
            if item["is_negative"]:
                negative += 1
            else:
                positive += 1
            correct = (not item["is_negative"] and item["verdict"] == "MET") or \
                      (item["is_negative"] and item["verdict"] == "UNMET")
            if correct:
                correctly_handled += 1

    normalized_score = compute_normalized_score(all_items)
    pass_rate = round((correctly_handled / total) * 100, 2) if total else 0.0
    per_category = {cat: compute_category_summary(items) for cat, items in results.items()}

    return {
        "total_criteria": total,
        "positive_criteria": positive,
        "negative_criteria": negative,
        "correctly_handled": correctly_handled,
        "pass_rate": pass_rate,
        "normalized_score": normalized_score,
        "per_category": per_category,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a research report against rubric criteria.")
    parser.add_argument("--report", type=Path, required=True, help="Path to report markdown file")
    parser.add_argument("--task", type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (optional)")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model that generated the report (e.g. 'openai:gpt-4.1', 'openai:gpt-4.1-mini', 'google_vertexai:gemini-2.5-pro'). "
             "Controls which evaluations_<slug>/ folder outputs go to."
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"Report not found: {args.report}")
    if not args.task.exists():
        raise FileNotFoundError(f"Task not found: {args.task}")

    # Resolve default output path
    if args.output is None:
        slug = _model_slug(args.model)
        output_dir = DRACO_DIR / f"evaluations_{slug}"
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

    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Total criteria:     {summary['total_criteria']}")
    print(f"Correctly handled:  {summary['correctly_handled']}")
    print(f"Pass rate:          {summary['pass_rate']}%")
    print(f"Normalized score:   {summary['normalized_score']}%")
    print(f"\nPer-category breakdown:")
    for cat, s in summary["per_category"].items():
        print(f"  {cat:<38} {s['correctly_handled']}/{s['total']}  score={s['normalized_score']}%")
    print(f"\nSaved to:           {args.output}")


if __name__ == "__main__":
    main()
