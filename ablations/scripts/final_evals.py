"""Final evaluation summary for GPT-4.1 and GPT-4.1-mini across Turn-1, Turn-2, and Turn-3.

Reports per model × turn:
  - Avg normalized score and avg pass rate (overall + per rubric category)
  - Incorporation rate: fraction of criteria unsat in vN that become sat in vN+1
  - Regression rate:    fraction of criteria sat in vN that become unsat in vN+1
  (Both rates computed with pooled counts across all tasks, overall and per category.)
  - Domain-level breakdown of all the above.

Turn-3 fallback: if a task has no v3_eval.json (achieved near-perfect in turn-2),
its v2 result is used as the turn-3 result. A note is printed per model.

Usage:
    python ablations/scripts/final_evals.py
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

ABLATIONS_DIR = Path(__file__).parent.parent

MODELS_GPT = [
    ("GPT-4.1",      "gpt4.1"),
    ("GPT-4.1-mini", "gpt4.1mini"),
]

MODELS_DEEPSEEK = [
    ("DeepSeek v4 Flash", "deepseekv4flash"),
]
TURNS = ["1", "2", "3"]

CATEGORIES = [
    ("factual-accuracy",              "FA"),
    ("breadth-and-depth-of-analysis", "BD"),
    ("presentation-quality",          "PQ"),
    ("citation-quality",              "CQ"),
]

CAT_KEY_TO_SHORT = {k: s for k, s in CATEGORIES}


# ---------------------------------------------------------------------------
# Task → domain mapping
# ---------------------------------------------------------------------------

def load_task_domains(tasks_dir: Path) -> dict[str, str]:
    mapping = {}
    for f in tasks_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            mapping[d["task_id"]] = d["domain"]
        except Exception:
            pass
    return mapping


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def fmt(v) -> str:
    return f"{v:.2f}%" if v is not None else "N/A"


def is_satisfied(item: dict) -> bool:
    """A criterion is satisfied if MET (positive) or UNMET (negative)."""
    return (not item["is_negative"] and item["verdict"] == "MET") or \
           (item["is_negative"] and item["verdict"] == "UNMET")


# ---------------------------------------------------------------------------
# per-turn score/pass-rate stats
# ---------------------------------------------------------------------------

def load_turn_summaries(evals_dir: Path, turn: str) -> list[tuple]:
    """Return list of (task_id, summary) for all tasks in a given turn."""
    summaries = []
    for f in sorted(evals_dir.glob(f"*_v{turn}_eval.json")):
        data = json.loads(f.read_text())
        summaries.append((data["task_id"], data.get("summary", {})))
    return summaries


def load_turn_summaries_with_fallback(evals_dir: Path, turn: str, fallback_turn: str) -> tuple[list[tuple], list[str]]:
    """Load turn summaries, falling back to fallback_turn for tasks with no v{turn} file.

    Returns:
        summaries: list of (task_id, summary)
        fallback_tasks: list of task_ids that used the fallback
    """
    # Load all v{turn} files that exist
    v_turn_tasks: dict[str, dict] = {}
    for f in sorted(evals_dir.glob(f"*_v{turn}_eval.json")):
        data = json.loads(f.read_text())
        v_turn_tasks[data["task_id"]] = data

    # Load all fallback (v{fallback_turn}) files
    v_fallback_tasks: dict[str, dict] = {}
    for f in sorted(evals_dir.glob(f"*_v{fallback_turn}_eval.json")):
        data = json.loads(f.read_text())
        v_fallback_tasks[data["task_id"]] = data

    summaries = []
    fallback_tasks = []

    for task_id in sorted(v_fallback_tasks):
        if task_id in v_turn_tasks:
            summaries.append((task_id, v_turn_tasks[task_id].get("summary", {})))
        else:
            summaries.append((task_id, v_fallback_tasks[task_id].get("summary", {})))
            fallback_tasks.append(task_id)

    return summaries, fallback_tasks


def compute_turn_stats(summaries: list[tuple]) -> dict:
    overall_ns = avg([s.get("normalized_score") for _, s in summaries])
    overall_pr = avg([s.get("pass_rate") for _, s in summaries])

    cat_ns: dict[str, list] = defaultdict(list)
    cat_pr: dict[str, list] = defaultdict(list)
    for _, s in summaries:
        for cat_key, _ in CATEGORIES:
            cat_data = s.get("per_category", {}).get(cat_key, {})
            total = cat_data.get("total")
            correctly = cat_data.get("correctly_handled")
            ns = cat_data.get("normalized_score")
            cat_ns[cat_key].append(ns)
            if total and correctly is not None:
                cat_pr[cat_key].append(round(correctly / total * 100, 2))

    per_category = {}
    for cat_key, short in CATEGORIES:
        per_category[short] = {
            "avg_normalized_score": avg(cat_ns[cat_key]),
            "avg_pass_rate": avg(cat_pr[cat_key]),
        }

    return {
        "n_tasks": len(summaries),
        "avg_normalized_score": overall_ns,
        "avg_pass_rate": overall_pr,
        "per_category": per_category,
    }


def compute_turn_stats_by_domain(summaries: list[tuple], task_domains: dict) -> dict[str, dict]:
    """Group summaries by domain and compute turn stats per domain."""
    by_domain: dict[str, list] = defaultdict(list)
    for task_id, s in summaries:
        domain = task_domains.get(task_id, "Unknown")
        by_domain[domain].append((task_id, s))
    return {domain: compute_turn_stats(slist) for domain, slist in sorted(by_domain.items())}


# ---------------------------------------------------------------------------
# incorporation / regression rates
# ---------------------------------------------------------------------------

def load_task_evals(evals_dir: Path, turn: str) -> dict[str, dict]:
    """Return {task_id: full eval dict} for a given turn."""
    evals = {}
    for f in sorted(evals_dir.glob(f"*_v{turn}_eval.json")):
        data = json.loads(f.read_text())
        evals[data["task_id"]] = data
    return evals


def load_task_evals_with_fallback(evals_dir: Path, turn: str, fallback_turn: str) -> dict[str, dict]:
    """Return {task_id: full eval dict} for a given turn, falling back for missing tasks."""
    v_turn = load_task_evals(evals_dir, turn)
    v_fallback = load_task_evals(evals_dir, fallback_turn)
    merged = {}
    for task_id in v_fallback:
        merged[task_id] = v_turn.get(task_id, v_fallback[task_id])
    return merged


def _incorp_regression_from_pairs(shared_tasks: list, v_prev_evals: dict, v_next_evals: dict) -> dict:
    """Core computation: pooled incorp/regression over a set of tasks."""
    overall_unsat_prev  = 0
    overall_sat_prev    = 0
    overall_improved    = 0
    overall_regressed   = 0

    cat_unsat_prev:  dict[str, int] = defaultdict(int)
    cat_sat_prev:    dict[str, int] = defaultdict(int)
    cat_improved:    dict[str, int] = defaultdict(int)
    cat_regressed:   dict[str, int] = defaultdict(int)

    for task_id in shared_tasks:
        prev_results = v_prev_evals[task_id].get("results", {})
        next_results = v_next_evals[task_id].get("results", {})

        for cat_key, _ in CATEGORIES:
            prev_items = {item["id"]: item for item in prev_results.get(cat_key, [])}
            next_items = {item["id"]: item for item in next_results.get(cat_key, [])}
            shared_ids = set(prev_items) & set(next_items)

            for cid in shared_ids:
                sat_prev = is_satisfied(prev_items[cid])
                sat_next = is_satisfied(next_items[cid])

                if sat_prev:
                    overall_sat_prev += 1
                    cat_sat_prev[cat_key] += 1
                    if not sat_next:
                        overall_regressed += 1
                        cat_regressed[cat_key] += 1
                else:
                    overall_unsat_prev += 1
                    cat_unsat_prev[cat_key] += 1
                    if sat_next:
                        overall_improved += 1
                        cat_improved[cat_key] += 1

    def rate(num: int, den: int) -> float | None:
        return round(num / den * 100, 2) if den > 0 else None

    per_category = {}
    for cat_key, short in CATEGORIES:
        per_category[short] = {
            "total_criteria":     cat_unsat_prev[cat_key] + cat_sat_prev[cat_key],
            "incorporation_rate": rate(cat_improved[cat_key],  cat_unsat_prev[cat_key]),
            "regression_rate":    rate(cat_regressed[cat_key], cat_sat_prev[cat_key]),
            "improved":   cat_improved[cat_key],
            "unsat_prev": cat_unsat_prev[cat_key],
            "regressed":  cat_regressed[cat_key],
            "sat_prev":   cat_sat_prev[cat_key],
        }

    return {
        "n_tasks_compared": len(shared_tasks),
        "overall": {
            "total_criteria":     overall_unsat_prev + overall_sat_prev,
            "incorporation_rate": rate(overall_improved,  overall_unsat_prev),
            "regression_rate":    rate(overall_regressed, overall_sat_prev),
            "improved":   overall_improved,
            "unsat_prev": overall_unsat_prev,
            "regressed":  overall_regressed,
            "sat_prev":   overall_sat_prev,
        },
        "per_category": per_category,
    }


def compute_incorp_regression(evals_dir: Path, prev_turn: str, next_turn: str,
                               fallback_turn: str | None = None) -> dict:
    prev_evals = load_task_evals(evals_dir, prev_turn)
    if fallback_turn:
        next_evals = load_task_evals_with_fallback(evals_dir, next_turn, fallback_turn)
    else:
        next_evals = load_task_evals(evals_dir, next_turn)
    shared_tasks = sorted(set(prev_evals) & set(next_evals))
    return _incorp_regression_from_pairs(shared_tasks, prev_evals, next_evals)


def compute_incorp_regression_by_domain(evals_dir: Path, task_domains: dict,
                                         prev_turn: str, next_turn: str,
                                         fallback_turn: str | None = None) -> dict[str, dict]:
    """Compute incorp/regression rates broken down by domain."""
    prev_evals = load_task_evals(evals_dir, prev_turn)
    if fallback_turn:
        next_evals = load_task_evals_with_fallback(evals_dir, next_turn, fallback_turn)
    else:
        next_evals = load_task_evals(evals_dir, next_turn)
    shared_tasks = sorted(set(prev_evals) & set(next_evals))

    by_domain: dict[str, list] = defaultdict(list)
    for task_id in shared_tasks:
        domain = task_domains.get(task_id, "Unknown")
        by_domain[domain].append(task_id)

    return {
        domain: _incorp_regression_from_pairs(tasks, prev_evals, next_evals)
        for domain, tasks in sorted(by_domain.items())
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_turn_stats(label: str, stats: dict, fallback_note: str = "") -> None:
    n = stats["n_tasks"]
    note = f"  {fallback_note}" if fallback_note else ""
    print(f"\n  {label}  (n={n}){note}")
    print(f"  {'─'*52}")
    print(f"  {'Metric':<28} {'Norm Score':>12} {'Pass Rate':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Overall':<28} {fmt(stats['avg_normalized_score']):>12} {fmt(stats['avg_pass_rate']):>10}")
    print()
    for _, short in CATEGORIES:
        cat = stats["per_category"][short]
        print(f"  {short:<28} {fmt(cat['avg_normalized_score']):>12} {fmt(cat['avg_pass_rate']):>10}")


def print_incorp_regression(ir: dict, label: str) -> None:
    n = ir["n_tasks_compared"]
    ov = ir["overall"]
    print(f"\n  {label}  (n={n} paired tasks)")
    print(f"  {'─'*62}")
    print(f"  {'Metric':<28} {'Incorp Rate':>14} {'Regression Rate':>16}")
    print(f"  {'─'*62}")
    total_overall = ov["unsat_prev"] + ov["sat_prev"]
    print(f"  {'Overall':<28} {fmt(ov['incorporation_rate']):>14} {fmt(ov['regression_rate']):>16}   (total criteria: {total_overall})")
    print(f"  {'':28}   ({ov['improved']}/{ov['unsat_prev']} unsat→sat)  "
          f"({ov['regressed']}/{ov['sat_prev']} sat→unsat)")
    print()
    for _, short in CATEGORIES:
        cat = ir["per_category"][short]
        total = cat["unsat_prev"] + cat["sat_prev"]
        print(f"  {short:<28} {fmt(cat['incorporation_rate']):>14} {fmt(cat['regression_rate']):>16}   (total criteria: {total})")
        print(f"  {'':28}   ({cat['improved']}/{cat['unsat_prev']} unsat→sat)  "
              f"({cat['regressed']}/{cat['sat_prev']} sat→unsat)")


def print_domain_turn_stats(domain_stats: dict[str, dict], turn: str) -> None:
    print(f"\n  Domain breakdown — Turn-{turn}")
    print(f"  {'─'*72}")
    print(f"  {'Domain':<32} {'n':>4} {'Norm Score':>12} {'Pass Rate':>10}")
    print(f"  {'─'*72}")
    for domain, stats in sorted(domain_stats.items()):
        print(f"  {domain:<32} {stats['n_tasks']:>4} "
              f"{fmt(stats['avg_normalized_score']):>12} "
              f"{fmt(stats['avg_pass_rate']):>10}")


def print_domain_incorp_regression(domain_ir: dict[str, dict], label: str) -> None:
    print(f"\n  Domain breakdown — {label}")
    print(f"  {'─'*72}")
    print(f"  {'Domain':<32} {'n':>4} {'Incorp Rate':>14} {'Regression Rate':>16}")
    print(f"  {'─'*72}")
    for domain, ir in sorted(domain_ir.items()):
        ov = ir["overall"]
        print(f"  {domain:<32} {ir['n_tasks_compared']:>4} "
              f"{fmt(ov['incorporation_rate']):>14} "
              f"{fmt(ov['regression_rate']):>16}")


# ---------------------------------------------------------------------------
# Turn-1 vs Turn-3 delta comparison
# ---------------------------------------------------------------------------

def _delta_str(v: float | None) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def compute_turn_delta(summaries_a: list[tuple], summaries_b: list[tuple]) -> dict:
    """Compute per-metric deltas between two sets of summaries (b - a)."""
    map_a = {tid: s for tid, s in summaries_a}
    map_b = {tid: s for tid, s in summaries_b}
    shared = sorted(set(map_a) & set(map_b))

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    ns_a = _avg([map_a[t].get("normalized_score") for t in shared])
    ns_b = _avg([map_b[t].get("normalized_score") for t in shared])
    pr_a = _avg([map_a[t].get("pass_rate") for t in shared])
    pr_b = _avg([map_b[t].get("pass_rate") for t in shared])

    per_category: dict[str, dict] = {}
    for cat_key, short in CATEGORIES:
        ns_a_cat = _avg([map_a[t].get("per_category", {}).get(cat_key, {}).get("normalized_score") for t in shared])
        ns_b_cat = _avg([map_b[t].get("per_category", {}).get(cat_key, {}).get("normalized_score") for t in shared])

        def _pr_cat(m, t):
            cd = m[t].get("per_category", {}).get(cat_key, {})
            tot = cd.get("total")
            cor = cd.get("correctly_handled")
            return round(cor / tot * 100, 2) if tot and cor is not None else None

        pr_a_cat = _avg([_pr_cat(map_a, t) for t in shared])
        pr_b_cat = _avg([_pr_cat(map_b, t) for t in shared])

        per_category[short] = {
            "ns_t1": ns_a_cat, "ns_t3": ns_b_cat,
            "ns_delta": round(ns_b_cat - ns_a_cat, 2) if ns_a_cat is not None and ns_b_cat is not None else None,
            "pr_t1": pr_a_cat, "pr_t3": pr_b_cat,
            "pr_delta": round(pr_b_cat - pr_a_cat, 2) if pr_a_cat is not None and pr_b_cat is not None else None,
        }

    return {
        "n_tasks": len(shared),
        "ns_t1": ns_a, "ns_t3": ns_b,
        "ns_delta": round(ns_b - ns_a, 2) if ns_a is not None and ns_b is not None else None,
        "pr_t1": pr_a, "pr_t3": pr_b,
        "pr_delta": round(pr_b - pr_a, 2) if pr_a is not None and pr_b is not None else None,
        "per_category": per_category,
    }


def compute_turn_delta_by_domain(summaries_a: list[tuple], summaries_b: list[tuple],
                                  task_domains: dict) -> dict[str, dict]:
    map_a = {tid: s for tid, s in summaries_a}
    map_b = {tid: s for tid, s in summaries_b}
    shared = sorted(set(map_a) & set(map_b))

    by_domain: dict[str, list] = defaultdict(list)
    for tid in shared:
        by_domain[task_domains.get(tid, "Unknown")].append(tid)

    result = {}
    for domain, tasks in sorted(by_domain.items()):
        subs_a = [(t, map_a[t]) for t in tasks]
        subs_b = [(t, map_b[t]) for t in tasks]
        result[domain] = compute_turn_delta(subs_a, subs_b)
    return result


def print_turn_delta(label: str, delta: dict) -> None:
    n = delta["n_tasks"]
    print(f"\n  {label}  (n={n})")
    print(f"  {'─'*72}")
    print(f"  {'Metric':<28} {'T1 Score':>10} {'T3 Score':>10} {'Δ Score':>10} {'T1 Pass':>9} {'T3 Pass':>9} {'Δ Pass':>8}")
    print(f"  {'─'*72}")
    print(f"  {'Overall':<28} "
          f"{fmt(delta['ns_t1']):>10} {fmt(delta['ns_t3']):>10} {_delta_str(delta['ns_delta']):>10} "
          f"{fmt(delta['pr_t1']):>9} {fmt(delta['pr_t3']):>9} {_delta_str(delta['pr_delta']):>8}")
    print()
    for _, short in CATEGORIES:
        cat = delta["per_category"][short]
        print(f"  {short:<28} "
              f"{fmt(cat['ns_t1']):>10} {fmt(cat['ns_t3']):>10} {_delta_str(cat['ns_delta']):>10} "
              f"{fmt(cat['pr_t1']):>9} {fmt(cat['pr_t3']):>9} {_delta_str(cat['pr_delta']):>8}")


def print_all_turns_table(
    t1_stats: dict | None,
    t2_stats: dict | None,
    t3_stats: dict | None,
    t1_domain: dict | None,
    t2_domain: dict | None,
    t3_domain: dict | None,
) -> None:
    """Print a single consolidated table: norm score + pass rate for T1/T2/T3."""

    def _ns(stats, key="avg_normalized_score"):
        return fmt(stats[key]) if stats else "N/A"

    def _pr(stats, key="avg_pass_rate"):
        return fmt(stats[key]) if stats else "N/A"

    # --- overall + per-category ---
    print(f"\n  All-Turns Summary — Normalized Score & Pass Rate")
    W = 84
    print(f"  {'─'*W}")
    print(f"  {'Metric':<28} {'T1 NS':>8} {'T1 PR':>8} {'T2 NS':>8} {'T2 PR':>8} {'T3 NS':>8} {'T3 PR':>8}")
    print(f"  {'─'*W}")

    def _row(label, ns1, pr1, ns2, pr2, ns3, pr3):
        print(f"  {label:<28} {ns1:>8} {pr1:>8} {ns2:>8} {pr2:>8} {ns3:>8} {pr3:>8}")

    _row("Overall",
         _ns(t1_stats), _pr(t1_stats),
         _ns(t2_stats), _pr(t2_stats),
         _ns(t3_stats), _pr(t3_stats))
    print()
    for _, short in CATEGORIES:
        ns1 = fmt(t1_stats["per_category"][short]["avg_normalized_score"]) if t1_stats else "N/A"
        pr1 = fmt(t1_stats["per_category"][short]["avg_pass_rate"])        if t1_stats else "N/A"
        ns2 = fmt(t2_stats["per_category"][short]["avg_normalized_score"]) if t2_stats else "N/A"
        pr2 = fmt(t2_stats["per_category"][short]["avg_pass_rate"])        if t2_stats else "N/A"
        ns3 = fmt(t3_stats["per_category"][short]["avg_normalized_score"]) if t3_stats else "N/A"
        pr3 = fmt(t3_stats["per_category"][short]["avg_pass_rate"])        if t3_stats else "N/A"
        _row(short, ns1, pr1, ns2, pr2, ns3, pr3)

    # --- per-domain ---
    all_domains = sorted(
        set(t1_domain or {}) | set(t2_domain or {}) | set(t3_domain or {})
    )
    if not all_domains:
        return

    print(f"\n  All-Turns Summary — Domain Breakdown (Normalized Score | Pass Rate)")
    DW = 96
    print(f"  {'─'*DW}")
    print(f"  {'Domain':<32} {'n':>4} {'T1 NS':>8} {'T1 PR':>8} {'T2 NS':>8} {'T2 PR':>8} {'T3 NS':>8} {'T3 PR':>8}")
    print(f"  {'─'*DW}")
    for domain in all_domains:
        d1 = (t1_domain or {}).get(domain, {})
        d2 = (t2_domain or {}).get(domain, {})
        d3 = (t3_domain or {}).get(domain, {})
        n = max(d1.get("n_tasks", 0), d2.get("n_tasks", 0), d3.get("n_tasks", 0))
        ns1 = fmt(d1.get("avg_normalized_score")) if d1 else "N/A"
        pr1 = fmt(d1.get("avg_pass_rate"))        if d1 else "N/A"
        ns2 = fmt(d2.get("avg_normalized_score")) if d2 else "N/A"
        pr2 = fmt(d2.get("avg_pass_rate"))        if d2 else "N/A"
        ns3 = fmt(d3.get("avg_normalized_score")) if d3 else "N/A"
        pr3 = fmt(d3.get("avg_pass_rate"))        if d3 else "N/A"
        print(f"  {domain:<32} {n:>4} {ns1:>8} {pr1:>8} {ns2:>8} {pr2:>8} {ns3:>8} {pr3:>8}")


def print_domain_turn_delta(domain_deltas: dict[str, dict], label: str) -> None:
    print(f"\n  Domain breakdown — {label}")
    print(f"  {'─'*72}")
    print(f"  {'Domain':<32} {'n':>4} {'T1 Score':>10} {'T3 Score':>10} {'Δ Score':>10} {'Δ Pass':>8}")
    print(f"  {'─'*72}")
    for domain, d in sorted(domain_deltas.items()):
        print(f"  {domain:<32} {d['n_tasks']:>4} "
              f"{fmt(d['ns_t1']):>10} {fmt(d['ns_t3']):>10} "
              f"{_delta_str(d['ns_delta']):>10} {_delta_str(d['pr_delta']):>8}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation summary.")
    parser.add_argument("--deepseekv4flash", action="store_true",
                        help="Analyse DeepSeek v4 Flash results instead of GPT-4.1 / GPT-4.1-mini.")
    args = parser.parse_args()

    models = MODELS_DEEPSEEK if args.deepseekv4flash else MODELS_GPT

    tasks_dir    = ABLATIONS_DIR / "tasks"
    task_domains = load_task_domains(tasks_dir)

    print(f"\n{'='*60}")
    print("  FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")

    for model_name, slug in models:
        evals_dir    = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}"
        analysis_dir = ABLATIONS_DIR / "analysis" / f"analysis_{slug}"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        model_result: dict = {"model": model_name}
        domain_result: dict = {"model": model_name}

        # collect stats per turn for the consolidated table
        all_turn_stats:  dict[str, dict] = {}
        all_turn_domain: dict[str, dict] = {}

        # --- per-turn score/pass-rate stats ---
        for turn in TURNS:
            if turn == "3":
                summaries, fallback_tasks = load_turn_summaries_with_fallback(
                    evals_dir, turn, fallback_turn="2"
                )
                if not summaries:
                    print(f"\n  Turn-{turn}: no v2 files found in {evals_dir}, skipping.")
                    continue
                fallback_note = ""
                if fallback_tasks:
                    fallback_note = f"[v2 used as fallback for {len(fallback_tasks)} task(s): {', '.join(fallback_tasks)}]"
            else:
                summaries = load_turn_summaries(evals_dir, turn)
                if not summaries:
                    print(f"\n  Turn-{turn}: no files found in {evals_dir}")
                    continue
                fallback_tasks = []
                fallback_note = ""

            stats = compute_turn_stats(summaries)
            model_result[f"turn_{turn}"] = stats
            if turn == "3" and fallback_tasks:
                model_result[f"turn_{turn}"]["v2_fallback_tasks"] = fallback_tasks
            all_turn_stats[turn] = stats
            print_turn_stats(f"Turn-{turn}", stats, fallback_note)

            domain_stats = compute_turn_stats_by_domain(summaries, task_domains)
            domain_result[f"turn_{turn}"] = {
                domain: {
                    "n_tasks":              ds["n_tasks"],
                    "avg_normalized_score": ds["avg_normalized_score"],
                    "avg_pass_rate":        ds["avg_pass_rate"],
                }
                for domain, ds in domain_stats.items()
            }
            all_turn_domain[turn] = domain_result[f"turn_{turn}"]
            print_domain_turn_stats(domain_stats, turn)

        # --- consolidated all-turns table ---
        print(f"\n{'─'*60}")
        print(f"  ALL-TURNS CONSOLIDATED TABLE")
        print(f"{'─'*60}")
        print_all_turns_table(
            all_turn_stats.get("1"), all_turn_stats.get("2"), all_turn_stats.get("3"),
            all_turn_domain.get("1"), all_turn_domain.get("2"), all_turn_domain.get("3"),
        )

        # --- incorporation / regression (v1 → v2) ---
        v1_evals = load_task_evals(evals_dir, "1")
        v2_evals = load_task_evals(evals_dir, "2")
        if v1_evals and v2_evals:
            ir_v1_v2 = compute_incorp_regression(evals_dir, "1", "2")
            model_result["v1_to_v2"] = ir_v1_v2
            print_incorp_regression(ir_v1_v2, "Incorporation / Regression Rates (v1→v2)")

            domain_ir_v1_v2 = compute_incorp_regression_by_domain(evals_dir, task_domains, "1", "2")
            print_domain_incorp_regression(domain_ir_v1_v2, "Incorporation / Regression (v1→v2)")

        # --- incorporation / regression (v2 → v3, with fallback for missing v3) ---
        ir_v2_v3 = compute_incorp_regression(evals_dir, "2", "3", fallback_turn="2")
        if ir_v2_v3["n_tasks_compared"] > 0:
            model_result["v2_to_v3"] = ir_v2_v3
            print_incorp_regression(ir_v2_v3, "Incorporation / Regression Rates (v2→v3)")

            domain_ir_v2_v3 = compute_incorp_regression_by_domain(
                evals_dir, task_domains, "2", "3", fallback_turn="2"
            )
            print_domain_incorp_regression(domain_ir_v2_v3, "Incorporation / Regression (v2→v3)")

        # --- Turn-1 vs Turn-3 delta comparison ---
        t1_summaries = load_turn_summaries(evals_dir, "1")
        t3_summaries, _ = load_turn_summaries_with_fallback(evals_dir, "3", fallback_turn="2")
        if t1_summaries and t3_summaries:
            print(f"\n{'─'*60}")
            print(f"  TURN-1 vs TURN-3 COMPARISON")
            print(f"{'─'*60}")

            delta = compute_turn_delta(t1_summaries, t3_summaries)
            model_result["t1_vs_t3_delta"] = delta
            print_turn_delta("Score / Pass-Rate Delta (T1 → T3)", delta)

            domain_deltas = compute_turn_delta_by_domain(t1_summaries, t3_summaries, task_domains)
            domain_result["t1_vs_t3_delta"] = {
                domain: {"n_tasks": d["n_tasks"], "ns_delta": d["ns_delta"], "pr_delta": d["pr_delta"]}
                for domain, d in domain_deltas.items()
            }
            print_domain_turn_delta(domain_deltas, "Score / Pass-Rate Delta (T1 → T3)")

            ir_v1_v3 = compute_incorp_regression(evals_dir, "1", "3", fallback_turn="2")
            model_result["v1_to_v3"] = ir_v1_v3
            print_incorp_regression(ir_v1_v3, "Incorporation / Regression Rates (v1→v3)")

            domain_ir_v1_v3 = compute_incorp_regression_by_domain(
                evals_dir, task_domains, "1", "3", fallback_turn="2"
            )
            print_domain_incorp_regression(domain_ir_v1_v3, "Incorporation / Regression (v1→v3)")

        # save overall summary
        out_path = analysis_dir / f"{slug}_final_summary.json"
        out_path.write_text(json.dumps(model_result, indent=2, ensure_ascii=False))
        print(f"\n  [Saved] {out_path}")

        # save domain summary
        domain_out = analysis_dir / f"{slug}_domain_summary.json"
        domain_out.write_text(json.dumps(domain_result, indent=2, ensure_ascii=False))
        print(f"  [Saved] {domain_out}")

    print()


if __name__ == "__main__":
    main()
