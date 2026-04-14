"""Compare two evaluation JSON files (e.g. turn-1 vs turn-2) and save an analysis.

Usage:
    python draco_eval/scripts/compare_turns.py \
        --v1 draco_eval/evaluations/task_002_v1_eval.json \
        --v2 draco_eval/evaluations/task_002_v2_eval.json \
        --task draco_eval/tasks/task_002.json

Output: draco_eval/analysis/<task_id>.json
"""

import argparse
import json
from pathlib import Path

DRACO_DIR = Path(__file__).parent.parent
RUBRIC_CATEGORIES = [
    "factual-accuracy",
    "breadth-and-depth-of-analysis",
    "presentation-quality",
    "citation-quality",
]


def is_pass(item: dict) -> bool:
    """A criterion is correctly handled if MET (positive) or UNMET (negative)."""
    return (not item["is_negative"] and item["verdict"] == "MET") or \
           (item["is_negative"] and item["verdict"] == "UNMET")


def analyse(v1_eval: dict, v2_eval: dict) -> dict:
    categories = {}
    all_regressions = []
    all_improvements = []
    all_neg_regressions = []

    for cat in RUBRIC_CATEGORIES:
        v1_items = {i["id"]: i for i in v1_eval["results"].get(cat, [])}
        v2_items = {i["id"]: i for i in v2_eval["results"].get(cat, [])}

        all_ids = list(v1_items.keys())  # both evals cover the same criteria

        v1_passed = [cid for cid in all_ids if is_pass(v1_items[cid])]
        v2_passed = [cid for cid in all_ids if is_pass(v2_items[cid])]

        v1_pass_set = set(v1_passed)
        v2_pass_set = set(v2_passed)

        regressions = sorted(v1_pass_set - v2_pass_set)   # passed v1, failed v2
        improvements = sorted(v2_pass_set - v1_pass_set)  # failed v1, passed v2
        stable_pass = sorted(v1_pass_set & v2_pass_set)
        stable_fail = sorted(set(all_ids) - v1_pass_set - v2_pass_set)

        # Negative-specific regressions: UNMET (good) in v1 → MET (bad) in v2
        neg_regressions = sorted(
            cid for cid in regressions if v1_items[cid]["is_negative"]
        )

        all_regressions.extend(regressions)
        all_improvements.extend(improvements)
        all_neg_regressions.extend(neg_regressions)

        # per-criterion detail
        criteria_detail = []
        for cid in all_ids:
            v1i = v1_items[cid]
            v2i = v2_items[cid]
            v1p = is_pass(v1i)
            v2p = is_pass(v2i)
            if v1p and v2p:
                status = "stable_pass"
            elif not v1p and not v2p:
                status = "stable_fail"
            elif not v1p and v2p:
                status = "improved"
            else:
                status = "regressed"

            criteria_detail.append({
                "id": cid,
                "weight": v1i["weight"],
                "is_negative": v1i["is_negative"],
                "v1_verdict": v1i["verdict"],
                "v2_verdict": v2i["verdict"],
                "status": status,
                "v1_explanation": v1i["explanation"],
                "v2_explanation": v2i["explanation"],
            })

        n = len(all_ids)
        v1_n = len(v1_passed)
        v2_n = len(v2_passed)
        reg_base = len(v1_passed)  # denominator for regression rate = how many passed in v1

        categories[cat] = {
            "total_criteria": n,
            "v1": {
                "passed": v1_n,
                "total": n,
                "pass_rate": round(v1_n / n, 4) if n else 0.0,
            },
            "v2": {
                "passed": v2_n,
                "total": n,
                "pass_rate": round(v2_n / n, 4) if n else 0.0,
            },
            "delta_passed": v2_n - v1_n,
            "regressions": {
                "count": len(regressions),
                "rate": round(len(regressions) / reg_base, 4) if reg_base else 0.0,
                "ids": regressions,
                "negative_introduced": {
                    "count": len(neg_regressions),
                    "ids": neg_regressions,
                },
            },
            "improvements": {
                "count": len(improvements),
                "ids": improvements,
            },
            "stable_pass": stable_pass,
            "stable_fail": stable_fail,
            "criteria": criteria_detail,
        }

    # overall
    total = sum(c["total_criteria"] for c in categories.values())
    v1_total_passed = sum(c["v1"]["passed"] for c in categories.values())
    v2_total_passed = sum(c["v2"]["passed"] for c in categories.values())
    overall_reg_base = v1_total_passed

    return {
        "task_id": v1_eval["task_id"],
        "domain": v1_eval.get("domain", ""),
        "v1_report": v1_eval["report_file"],
        "v2_report": v2_eval["report_file"],
        "overall": {
            "total_criteria": total,
            "v1": {
                "passed": v1_total_passed,
                "total": total,
                "pass_rate": round(v1_total_passed / total, 4),
                "normalized_score": v1_eval["summary"]["normalized_score"],
            },
            "v2": {
                "passed": v2_total_passed,
                "total": total,
                "pass_rate": round(v2_total_passed / total, 4),
                "normalized_score": v2_eval["summary"]["normalized_score"],
            },
            "delta_passed": v2_total_passed - v1_total_passed,
            "delta_normalized_score": round(
                v2_eval["summary"]["normalized_score"] - v1_eval["summary"]["normalized_score"], 2
            ),
            "regressions": {
                "count": len(all_regressions),
                "rate": round(len(all_regressions) / overall_reg_base, 4) if overall_reg_base else 0.0,
                "ids": sorted(all_regressions),
                "negative_introduced": {
                    "count": len(all_neg_regressions),
                    "ids": sorted(all_neg_regressions),
                },
            },
            "improvements": {
                "count": len(all_improvements),
                "ids": sorted(all_improvements),
            },
        },
        "by_category": categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two turn evaluations.")
    parser.add_argument("--v1", type=Path, required=True, help="Turn-1 eval JSON")
    parser.add_argument("--v2", type=Path, required=True, help="Turn-2 eval JSON")
    parser.add_argument("--task", type=Path, required=True, help="Task JSON (for task_id)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    v1_eval = json.loads(args.v1.read_text())
    v2_eval = json.loads(args.v2.read_text())
    task = json.loads(args.task.read_text())
    task_id = task["task_id"]

    result = analyse(v1_eval, v2_eval)

    if args.output is None:
        out_dir = DRACO_DIR / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{task_id}.json"

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    ov = result["overall"]
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"  v1: {ov['v1']['passed']}/{ov['v1']['total']} passed  (score {ov['v1']['normalized_score']}%)")
    print(f"  v2: {ov['v2']['passed']}/{ov['v2']['total']} passed  (score {ov['v2']['normalized_score']}%)")
    print(f"  Δ:  +{ov['delta_passed']} criteria  (+{ov['delta_normalized_score']} pts)")
    print(f"\nRegressions: {ov['regressions']['count']}  ({ov['regressions']['rate']*100:.1f}% of v1 passes)")
    for cid in ov['regressions']['ids']:
        print(f"  - {cid}")
    print(f"\nImprovements: {ov['improvements']['count']}")
    print(f"\nBy category:")
    for cat, data in result["by_category"].items():
        print(f"  {cat:<40} v1 {data['v1']['passed']}/{data['v1']['total']}  →  v2 {data['v2']['passed']}/{data['v2']['total']}  (regressions: {data['regressions']['count']})")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
