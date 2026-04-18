"""Summarize evaluation results from draco_eval/evaluations_<model_slug>/.

Reads from evaluations_<slug>/ and writes summary to analysis_<slug>/.

Usage:
    # GPT-4.1 turn-1 (default)
    python draco_eval/scripts/summarize_evals.py

    # GPT-4.1 turn-2
    python draco_eval/scripts/summarize_evals.py --turn 2

    # GPT-4.1 turn-1 vs turn-2 comparison
    python draco_eval/scripts/summarize_evals.py --turn 1 --compare-turn 2

    # GPT-4.1-mini
    python draco_eval/scripts/summarize_evals.py --model openai:gpt-4.1-mini

    # Gemini 2.5 Pro
    python draco_eval/scripts/summarize_evals.py --model google_vertexai:gemini-2.5-pro
    python draco_eval/scripts/summarize_evals.py --turn 1 --compare-turn 2 --model google_vertexai:gemini-2.5-pro
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

DRACO_DIR = Path(__file__).parent.parent
TASKS_DIR = DRACO_DIR / "tasks"


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")

CATEGORIES = [
    "factual-accuracy",
    "breadth-and-depth-of-analysis",
    "presentation-quality",
    "citation-quality",
]

CAT_SHORT = {
    "factual-accuracy": "Factual Accuracy",
    "breadth-and-depth-of-analysis": "Breadth/Depth",
    "presentation-quality": "Presentation",
    "citation-quality": "Citation Quality",
}


def load_domain_map():
    domain_map = {}
    for f in TASKS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            domain_map[d["task_id"]] = d.get("domain", "Unknown")
        except Exception:
            pass
    return domain_map


def load_turn_data(turn, evals_dir):
    pattern = f"*_v{turn}_eval.json"
    eval_files = sorted(evals_dir.glob(pattern))
    results = {}
    for f in eval_files:
        data = json.loads(f.read_text())
        task_id = data.get("task_id", f.stem)
        summary = data.get("summary", {})
        results[task_id] = {
            "normalized_score": summary.get("normalized_score"),
            "pass_rate": summary.get("pass_rate"),
            "per_category": {
                cat: summary.get("per_category", {}).get(cat, {})
                for cat in CATEGORIES
            },
        }
    return eval_files, results


def print_domain_analysis(turn_label, data, domain_map, compare_data=None, compare_label=None):
    domain_scores = defaultdict(list)
    domain_cat_scores = defaultdict(lambda: defaultdict(list))

    for task_id, task_data in data.items():
        domain = domain_map.get(task_id, "Unknown")
        ns = task_data["normalized_score"]
        if ns is not None:
            domain_scores[domain].append(ns)
        for cat in CATEGORIES:
            cat_ns = task_data["per_category"].get(cat, {}).get("normalized_score")
            if cat_ns is not None:
                domain_cat_scores[domain][cat].append(cat_ns)

    compare_domain_scores = defaultdict(list)
    compare_domain_cat_scores = defaultdict(lambda: defaultdict(list))
    if compare_data:
        for task_id, task_data in compare_data.items():
            domain = domain_map.get(task_id, "Unknown")
            ns = task_data["normalized_score"]
            if ns is not None:
                compare_domain_scores[domain].append(ns)
            for cat in CATEGORIES:
                cat_ns = task_data["per_category"].get(cat, {}).get("normalized_score")
                if cat_ns is not None:
                    compare_domain_cat_scores[domain][cat].append(cat_ns)

    all_domains = sorted(set(domain_scores.keys()) | set(compare_domain_scores.keys()))

    print(f"\n{'='*70}")
    if compare_data:
        print(f"DOMAIN-WISE ANALYSIS ({turn_label} vs {compare_label})")
        print(f"{'='*70}")
        print(f"\n  {'Domain':<32} {'n':>3} {turn_label+' Avg':>10} {compare_label+' Avg':>10} {'Delta':>8}")
        print(f"  {'-'*65}")
        domain_summary = {}
        for domain in all_domains:
            n = max(len(domain_scores[domain]), len(compare_domain_scores[domain]))
            avg1 = round(sum(domain_scores[domain]) / len(domain_scores[domain]), 2) if domain_scores[domain] else None
            avg2 = round(sum(compare_domain_scores[domain]) / len(compare_domain_scores[domain]), 2) if compare_domain_scores[domain] else None
            delta = round(avg2 - avg1, 2) if avg1 is not None and avg2 is not None else None
            delta_str = f"{'+'if delta >= 0 else ''}{delta}%" if delta is not None else "N/A"
            a1_str = f"{avg1}%" if avg1 is not None else "N/A"
            a2_str = f"{avg2}%" if avg2 is not None else "N/A"
            print(f"  {domain:<32} {n:>3} {a1_str:>10} {a2_str:>10} {delta_str:>8}")
            domain_summary[domain] = {"avg_t1": avg1, "avg_t2": avg2, "delta": delta, "n": n}
    else:
        print(f"DOMAIN-WISE ANALYSIS ({turn_label})")
        print(f"{'='*70}")
        print(f"\n  {'Domain':<32} {'n':>3} {'Avg Score':>10}")
        print(f"  {'-'*50}")
        domain_summary = {}
        for domain in all_domains:
            n = len(domain_scores[domain])
            avg = round(sum(domain_scores[domain]) / n, 2) if domain_scores[domain] else None
            avg_str = f"{avg}%" if avg is not None else "N/A"
            print(f"  {domain:<32} {n:>3} {avg_str:>10}")
            domain_summary[domain] = {"avg": avg, "n": n}

    print(f"\n  DOMAIN × CATEGORY BREAKDOWN:")
    for domain in all_domains:
        print(f"\n  [{domain}]")
        if compare_data:
            print(f"  {'Category':<24} {turn_label:>8} {compare_label:>8} {'Delta':>8}")
            print(f"  {'-'*52}")
            for cat in CATEGORIES:
                c1 = domain_cat_scores[domain][cat]
                c2 = compare_domain_cat_scores[domain][cat]
                a1 = round(sum(c1) / len(c1), 2) if c1 else None
                a2 = round(sum(c2) / len(c2), 2) if c2 else None
                delta = round(a2 - a1, 2) if a1 is not None and a2 is not None else None
                delta_str = f"{'+'if delta >= 0 else ''}{delta}%" if delta is not None else "N/A"
                a1_str = f"{a1}%" if a1 is not None else "N/A"
                a2_str = f"{a2}%" if a2 is not None else "N/A"
                print(f"  {CAT_SHORT[cat]:<24} {a1_str:>8} {a2_str:>8} {delta_str:>8}")
        else:
            print(f"  {'Category':<24} {'Avg Score':>10}")
            print(f"  {'-'*36}")
            for cat in CATEGORIES:
                c1 = domain_cat_scores[domain][cat]
                avg = round(sum(c1) / len(c1), 2) if c1 else None
                avg_str = f"{avg}%" if avg is not None else "N/A"
                print(f"  {CAT_SHORT[cat]:<24} {avg_str:>10}")

    return domain_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", default="1", choices=["1", "2"])
    parser.add_argument("--compare-turn", choices=["1", "2"], default=None,
                        help="Optional second turn to compare against")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model being summarised. Controls which evaluations_<slug>/ and analysis_<slug>/ folders are used."
    )
    args = parser.parse_args()

    slug = _model_slug(args.model)
    EVALS_DIR = DRACO_DIR / f"evaluations_{slug}"
    ANALYSIS_DIR = DRACO_DIR / f"analysis_{slug}"

    domain_map = load_domain_map()

    eval_files, data = load_turn_data(args.turn, EVALS_DIR)

    if not eval_files:
        print(f"No evaluation files found matching: {EVALS_DIR / f'*_v{args.turn}_eval.json'}")
        return

    compare_files, compare_data = None, None
    if args.compare_turn and args.compare_turn != args.turn:
        compare_files, compare_data = load_turn_data(args.compare_turn, EVALS_DIR)
        if not compare_files:
            print(f"No evaluation files found for compare turn {args.compare_turn}, skipping comparison.")
            compare_data = None

    normalized_scores = []
    pass_rates = []
    cat_scores = defaultdict(list)
    cat_correctly = defaultdict(list)
    cat_totals = defaultdict(list)
    per_task = {}

    all_task_ids = sorted(set(data.keys()) | (set(compare_data.keys()) if compare_data else set()))

    if compare_data:
        tl = f"T{args.turn}"
        cl = f"T{args.compare_turn}"
        print(f"\nEvaluations found: T{args.turn}={len(eval_files)}  T{args.compare_turn}={len(compare_files or [])}\n")
        print(f"{'Task':<15} {'Domain':<32} {tl+' Score':>10} {cl+' Score':>10} {tl+' Pass':>9} {cl+' Pass':>9} {'Delta':>8}")
        print("-" * 97)
    else:
        print(f"\nTurn-{args.turn} evaluations found: {len(eval_files)}\n")
        print(f"{'Task':<15} {'Domain':<35} {'Norm Score':>12} {'Pass Rate':>10}")
        print("-" * 75)

    for task_id in all_task_ids:
        task_data = data.get(task_id, {})
        ns = task_data.get("normalized_score")
        pr = task_data.get("pass_rate")
        domain = domain_map.get(task_id, "?")

        if ns is not None:
            normalized_scores.append(ns)
        if pr is not None:
            pass_rates.append(pr)

        per_task[task_id] = {"normalized_score": ns, "pass_rate": pr, "domain": domain}

        if compare_data:
            cmp = compare_data.get(task_id, {})
            ns2 = cmp.get("normalized_score")
            pr2 = cmp.get("pass_rate")
            delta = round(ns2 - ns, 2) if ns is not None and ns2 is not None else None
            delta_str = f"{'+'if delta >= 0 else ''}{delta}%" if delta is not None else "N/A"
            print(f"{task_id:<15} {domain:<32} {str(ns)+'%':>10} {str(ns2)+'%':>10} {str(pr)+'%':>9} {str(pr2)+'%':>9} {delta_str:>8}")
        else:
            print(f"{task_id:<15} {domain:<35} {str(ns) + '%':>12} {str(pr) + '%':>10}")

        for cat in CATEGORIES:
            cat_data = task_data.get("per_category", {}).get(cat, {})
            if cat_data:
                cat_scores[cat].append(cat_data.get("normalized_score", 0))
                cat_correctly[cat].append(cat_data.get("correctly_handled", 0))
                cat_totals[cat].append(cat_data.get("total", 0))

    n = len(normalized_scores)
    avg_norm = round(sum(normalized_scores) / n, 2)
    avg_pass = round(sum(pass_rates) / n, 2)

    if compare_data:
        cmp_scores = [v["normalized_score"] for v in compare_data.values() if v.get("normalized_score") is not None]
        cmp_passes = [v["pass_rate"] for v in compare_data.values() if v.get("pass_rate") is not None]
        avg_norm2 = round(sum(cmp_scores) / len(cmp_scores), 2) if cmp_scores else None
        avg_pass2 = round(sum(cmp_passes) / len(cmp_passes), 2) if cmp_passes else None
        delta_norm = round(avg_norm2 - avg_norm, 2) if avg_norm2 is not None else None
        delta_pass = round(avg_pass2 - avg_pass, 2) if avg_pass2 is not None else None

        print(f"\n{'='*65}")
        print(f"OVERALL SUMMARY (n={n} tasks, T{args.turn} vs T{args.compare_turn})")
        print(f"{'='*65}")
        print(f"  {'Metric':<28} {'T'+args.turn:>8} {'T'+args.compare_turn:>8} {'Delta':>8}")
        print(f"  {'-'*55}")
        d_norm = f"{'+'if delta_norm >= 0 else ''}{delta_norm}%" if delta_norm is not None else "N/A"
        d_pass = f"{'+'if delta_pass >= 0 else ''}{delta_pass}%" if delta_pass is not None else "N/A"
        print(f"  {'Avg normalized score':<28} {avg_norm:>7}% {avg_norm2:>7}% {d_norm:>8}")
        print(f"  {'Avg pass rate':<28} {avg_pass:>7}% {avg_pass2:>7}% {d_pass:>8}")
    else:
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

    turn_label = f"T{args.turn}"
    compare_label = f"T{args.compare_turn}" if args.compare_turn else None
    domain_summary = print_domain_analysis(turn_label, data, domain_map, compare_data, compare_label)

    result = {
        "turn": args.turn,
        "n_tasks": n,
        "avg_normalized_score": avg_norm,
        "avg_pass_rate": avg_pass,
        "per_category": per_category_summary,
        "per_domain": domain_summary,
        "per_task": per_task,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    out_path = ANALYSIS_DIR / f"overall_summary_v{args.turn}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
