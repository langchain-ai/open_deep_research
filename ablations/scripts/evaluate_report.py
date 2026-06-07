"""Evaluate a research report against a task's rubric criteria.

Output goes to ablations/evaluations/evaluations_<model_slug>[_<save_name>]/ by default.
Files evaluated by the Gemini judge get a "GEM_" prefix in their filename.

Usage:
    # GPT-4.1 (default, OpenAI judge)
    python ablations/scripts/evaluate_report.py \
        --report ablations/reports/reports_gpt4.1/task_002_v1.md \
        --task   ablations/tasks/task_002.json

    # GPT-4.1-mini (OpenAI judge)
    python ablations/scripts/evaluate_report.py \
        --report ablations/reports/reports_gpt4.1mini/task_002_v1.md \
        --task   ablations/tasks/task_002.json \
        --model  openai:gpt-4.1-mini

    # GPT-4.1 with Gemini 3 Pro judge
    python ablations/scripts/evaluate_report.py \
        --report ablations/reports/reports_gpt4.1/task_002_v1.md \
        --task   ablations/tasks/task_002.json \
        --judge  gemini

    # With save_name suffix
    python ablations/scripts/evaluate_report.py \
        --report    ablations/reports/reports_gpt4.1_ablation_1/task_002_v1.md \
        --task      ablations/tasks/task_002.json \
        --save_name ablation_1

    # Override output path explicitly
    python ablations/scripts/evaluate_report.py \
        --report ablations/reports/reports_gpt4.1/task_039_v2.md \
        --task   ablations/tasks/task_039.json \
        --output ablations/evaluations/evaluations_gpt4.1/task_039_v2_eval.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_JUDGE_MODEL = "gpt-5.2-2025-12-11"
GEMINI_JUDGE_MODEL = "gemini-3.1-pro-preview"
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
    parts = text.split("=== USER PROMPT ===")
    user_template = parts[1].strip()
    system_part = parts[0].split("=== SYSTEM PROMPT ===")[1].strip()
    return system_part, user_template


def _parse_verdict(parsed: dict, raw: str, source: str, skip_fails: bool = False) -> dict:
    """Extract and validate verdict/explanation from a parsed JSON response.

    If skip_fails=True, any verdict that is not 'MET' or 'UNMET' is coerced to
    'UNMET' instead of raising, so a single malformed judge response never
    aborts the whole evaluation run.
    """
    verdict = (
        parsed.get("criterion_status")
        or parsed.get("status")
        or parsed.get("verdict")
    )
    if verdict not in ("MET", "UNMET"):
        if skip_fails:
            print(f" [skip_fails: coercing unexpected verdict '{verdict}' → UNMET]", flush=True)
            verdict = "UNMET"
        else:
            raise ValueError(f"Unexpected verdict '{verdict}' from {source}. Raw: {raw}")
    return {"verdict": verdict, "explanation": parsed.get("explanation", "")}


def evaluate_criterion(
    client: OpenAI,
    system_prompt: str,
    user_template: str,
    criterion_type: str,
    criterion_requirement: str,
    original_query: str,
    report_text: str,
    skip_fails: bool = False,
) -> dict:
    """Call OpenAI to evaluate a single criterion. Returns {verdict, explanation}."""
    user_msg = user_template
    user_msg = user_msg.replace("{criterion_type}", criterion_type)
    user_msg = user_msg.replace("{criterion_requirement}", criterion_requirement)
    user_msg = user_msg.replace("{original_query}", original_query)
    user_msg = user_msg.replace("{report_text}", report_text)
    response = client.chat.completions.create(
        model=OPENAI_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        reasoning_effort="none",
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return _parse_verdict(json.loads(raw), raw, "OpenAI", skip_fails=skip_fails)


def evaluate_criterion_gemini(
    system_prompt: str,
    user_template: str,
    criterion_type: str,
    criterion_requirement: str,
    original_query: str,
    report_text: str,
    skip_fails: bool = False,
) -> dict:
    """Call Gemini judge to evaluate a single criterion. Returns {verdict, explanation}.

    Uses global Vertex AI endpoint, temperature=1.0, seed=42, thinking_level=LOW.
    """
    import time
    from google import genai as google_genai
    from google.genai import types as genai_types

    user_msg = user_template
    user_msg = user_msg.replace("{criterion_type}", criterion_type)
    user_msg = user_msg.replace("{criterion_requirement}", criterion_requirement)
    user_msg = user_msg.replace("{original_query}", original_query)
    user_msg = user_msg.replace("{report_text}", report_text)

    client = google_genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location="global",
    )

    for attempt in range(8):
        try:
            response = client.models.generate_content(
                model=GEMINI_JUDGE_MODEL,
                contents=user_msg,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
                    temperature=1.0,
                    seed=42,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text
            return _parse_verdict(json.loads(raw), raw, "Gemini", skip_fails=skip_fails)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "TransportError", "ConnectTimeout", "timed out")):
                wait = 30 * (2 ** attempt)
                print(f" [transient error ({err[:60].strip()}), retrying in {wait}s]", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini transient error persisted after 8 retries.")


def compute_normalized_score(items: list) -> float:
    """Compute normalized score per the paper's formula."""
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
        help="Agent model that generated the report. Controls which evaluations/evaluations_<slug>/ folder outputs go to."
    )
    parser.add_argument(
        "--save_name", default="",
        help="Optional suffix for output folders (e.g. 'ablation_1' → evaluations/evaluations_gpt4.1_ablation_1/)."
    )
    parser.add_argument(
        "--judge", default="openai", choices=["openai", "gemini"],
        help="Judge model to use: 'openai' (default, gpt-5.2) or 'gemini' (Gemini 3 Pro). "
             "Gemini-judged files get a 'GEM_' prefix in their filename."
    )
    parser.add_argument(
        "--skip_fails", action="store_true", default=False,
        help="If set, any verdict that is not MET or UNMET is coerced to UNMET "
             "instead of raising an error. Useful when the judge occasionally "
             "omits the verdict field from its JSON response."
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"Report not found: {args.report}")
    if not args.task.exists():
        raise FileNotFoundError(f"Task not found: {args.task}")

    # File prefix for Gemini-judged outputs
    file_prefix = "GEM_" if args.judge == "gemini" else ""

    if args.output is None:
        slug = _model_slug(args.model)
        suffix = f"_{args.save_name}" if args.save_name else ""
        folder_prefix = "GEM_" if args.judge == "gemini" else ""
        output_dir = DRACO_DIR / "evaluations" / f"{folder_prefix}evaluations_{slug}{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{args.report.stem}_eval.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    task = json.loads(args.task.read_text())
    report_text = args.report.read_text()
    system_prompt, user_template = load_judge_prompt(JUDGE_PROMPT_PATH)
    original_query = task["prompt"]

    # Set up judge client
    if args.judge == "gemini":
        openai_client = None
        print(f"Judge: {GEMINI_JUDGE_MODEL} (thinking_level=LOW, temperature=1.0, seed=42, location=global) → prefix GEM_")
    else:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Judge: OpenAI {OPENAI_JUDGE_MODEL}")

    results = {cat: [] for cat in RUBRIC_CATEGORIES}

    for category in RUBRIC_CATEGORIES:
        criteria = task.get("rubrics", {}).get(category, [])
        for criterion in criteria:
            cid = criterion["id"]
            requirement = criterion["requirement"]
            weight = criterion["weight"]
            is_negative = weight < 0
            criterion_type = "negative" if is_negative else "positive"

            print(f"  Evaluating [{category}] {cid} ...", end=" ", flush=True)

            if args.judge == "gemini":
                outcome = evaluate_criterion_gemini(
                    system_prompt=system_prompt,
                    user_template=user_template,
                    criterion_type=criterion_type,
                    criterion_requirement=requirement,
                    original_query=original_query,
                    report_text=report_text,
                    skip_fails=args.skip_fails,
                )
            else:
                outcome = evaluate_criterion(
                    client=openai_client,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    criterion_type=criterion_type,
                    criterion_requirement=requirement,
                    original_query=original_query,
                    report_text=report_text,
                    skip_fails=args.skip_fails,
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
