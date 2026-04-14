"""Summarize evaluation results from draco_eval/evaluations/.

Usage:
    python draco_eval/scripts/summarize_evals.py
    python draco_eval/scripts/summarize_evals.py --turn 2
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

DRACO_DIR = Path(__file__).parent.parent
EVALS_DIR = DRACO_DIR / "evaluations"
ANALYSIS_DIR = DRACO_DIR / "analysis"

CATEGORIES = [
    "factual-accuracy",
    "breadth-and-depth-of-analysis",
    "presentation-quality",
    "citation-quality",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", default="1", choices=["1", "2"])
    args = parser.parse_args()

    pattern = f"*_v{args.turn}_eval.json"
    eval_files = sorted(EVALS_DIR.glob(pattern))

    if not eval_files:
        print(f"No evaluation files found matching: {EVALS_DIR / pattern}")
        return

    normalized_scores = []
    pass_rates = []
    cat_scores = defaultdict(list)
    cat_correctly = defaultdict(list)
    cat_totals = defaultdict(list)
    per_task = {}

    print(f"\nTurn-{args.turn} evaluations found: {len(eval_files)}\n")
    print(f"{'Task':<15} {'Norm Score':>12} {'Pass Rate':>10}")
    print("-" * 40)

    for f in eval_files:
        data = json.loads(f.read_text())
        summary = data.get("summary", {})
        task_id = data.get("task_id", f.stem)

        ns = summary.get("normalized_score")
        pr = summary.get("pass_rate")

        if ns is not None:
            normalized_scores.append(ns)
        if pr is not None:
            pass_rates.append(pr)

        per_task[task_id] = {"normalized_score": ns, "pass_rate": pr}
        print(f"{task_id:<15} {str(ns) + '%':>12} {str(pr) + '%':>10}")

        per_cat = summary.get("per_category", {})
        for cat in CATEGORIES:
            if cat in per_cat:
                cat_scores[cat].append(per_cat[cat].get("normalized_score", 0))
                cat_correctly[cat].append(per_cat[cat].get("correctly_handled", 0))
                cat_totals[cat].append(per_cat[cat].get("total", 0))

    n = len(normalized_scores)
    avg_norm = round(sum(normalized_scores) / n, 2)
    avg_pass = round(sum(pass_rates) / n, 2)

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY (n={n} tasks, turn {args.turn})")
    print(f"{'='*60}")
    print(f"  Avg normalized score : {avg_norm}%")
    print(f"  Avg pass rate        : {avg_pass}%")

    per_category_summary = {}
    print(f"\nPER-CATEGORY AVERAGES:")
    print(f"  {'Category':<40} {'Avg Score':>10} {'Avg Correct/Total':>20}")
    print(f"  {'-'*72}")
    for cat in CATEGORIES:
        if cat_scores[cat]:
            avg_score = round(sum(cat_scores[cat]) / len(cat_scores[cat]), 2)
            avg_correct = round(sum(cat_correctly[cat]) / len(cat_correctly[cat]), 2)
            avg_total = round(sum(cat_totals[cat]) / len(cat_totals[cat]), 2)
            per_category_summary[cat] = {
                "avg_normalized_score": avg_score,
                "avg_correctly_handled": avg_correct,
                "avg_total": avg_total,
            }
            print(f"  {cat:<40} {avg_score:>9.2f}%  {avg_correct:>6.1f} / {avg_total:.1f}")

    result = {
        "turn": args.turn,
        "n_tasks": n,
        "avg_normalized_score": avg_norm,
        "avg_pass_rate": avg_pass,
        "per_category": per_category_summary,
        "per_task": per_task,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"overall_summary_v{args.turn}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
