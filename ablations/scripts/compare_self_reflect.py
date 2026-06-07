"""Compare self-reflection baseline vs feedback-driven method (v2 evals).

For a given model, compares:
  - Self-reflection:      ablations/evaluations/evaluations_<slug>_self_reflect/  (v2 files)
  - Feedback method:      ablations/evaluations/evaluations_<slug>/                (v2 files)
  - Shared v1 baseline:   ablations/evaluations/evaluations_<slug>/                (v1 files)

Metrics reported overall and per domain:
  - Avg normalized score  (T1, self-reflect v2, feedback v2)
  - Avg pass rate         (T1, self-reflect v2, feedback v2)
  - Incorporation rate    (v1 → v2, for each method separately)
  - Regression rate       (v1 → v2, for each method separately)

Usage:
    # GPT-4.1-mini (default)
    python ablations/scripts/compare_self_reflect.py

    # GPT-4.1
    python ablations/scripts/compare_self_reflect.py --model openai:gpt-4.1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ABLATIONS_DIR = Path(__file__).parent.parent

CATEGORIES = [
    ("factual-accuracy",              "FA"),
    ("breadth-and-depth-of-analysis", "BD"),
    ("presentation-quality",          "PQ"),
    ("citation-quality",              "CQ"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _model_slug(model: str) -> str:
    return model.split(":")[-1].replace("-", "")


def avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def fmt(v) -> str:
    return f"{v:.2f}%" if v is not None else "  N/A  "


def delta_str(v: float | None) -> str:
    if v is None:
        return "  N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def is_satisfied(item: dict) -> bool:
    return (not item["is_negative"] and item["verdict"] == "MET") or \
           (item["is_negative"] and item["verdict"] == "UNMET")


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_evals(evals_dir: Path, turn: str) -> dict[str, dict]:
    """Return {task_id: full eval dict} for all *_v{turn}_eval.json in dir."""
    result = {}
    for f in sorted(evals_dir.glob(f"*_v{turn}_eval.json")):
        data = json.loads(f.read_text())
        result[data["task_id"]] = data
    return result


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
# score / pass-rate stats
# ---------------------------------------------------------------------------

def compute_score_stats(evals: dict[str, dict]) -> dict:
    """Overall + per-category avg normalized score and pass rate."""
    ns_list, pr_list = [], []
    cat_ns: dict[str, list] = defaultdict(list)
    cat_pr: dict[str, list] = defaultdict(list)

    for data in evals.values():
        s = data.get("summary", {})
        ns_list.append(s.get("normalized_score"))
        pr_list.append(s.get("pass_rate"))
        for cat_key, _ in CATEGORIES:
            cd = s.get("per_category", {}).get(cat_key, {})
            cat_ns[cat_key].append(cd.get("normalized_score"))
            tot = cd.get("total")
            cor = cd.get("correctly_handled")
            if tot and cor is not None:
                cat_pr[cat_key].append(round(cor / tot * 100, 2))

    per_category = {}
    for cat_key, short in CATEGORIES:
        per_category[short] = {
            "avg_ns": avg(cat_ns[cat_key]),
            "avg_pr": avg(cat_pr[cat_key]),
        }

    return {
        "n": len(evals),
        "avg_ns": avg(ns_list),
        "avg_pr": avg(pr_list),
        "per_category": per_category,
    }


def compute_score_stats_by_domain(evals: dict[str, dict], domains: dict[str, str]) -> dict[str, dict]:
    by_domain: dict[str, dict] = defaultdict(dict)
    for task_id, data in evals.items():
        domain = domains.get(task_id, "Unknown")
        by_domain[domain][task_id] = data
    return {d: compute_score_stats(sub) for d, sub in sorted(by_domain.items())}


# ---------------------------------------------------------------------------
# incorporation / regression
# ---------------------------------------------------------------------------

def compute_ir(v1_evals: dict, v2_evals: dict) -> dict:
    """Pooled incorporation + regression rates (overall + per category)."""
    shared = sorted(set(v1_evals) & set(v2_evals))

    overall_unsat, overall_sat, overall_imp, overall_reg = 0, 0, 0, 0
    cat_unsat: dict[str, int] = defaultdict(int)
    cat_sat:   dict[str, int] = defaultdict(int)
    cat_imp:   dict[str, int] = defaultdict(int)
    cat_reg:   dict[str, int] = defaultdict(int)

    for task_id in shared:
        r1 = v1_evals[task_id].get("results", {})
        r2 = v2_evals[task_id].get("results", {})
        for cat_key, _ in CATEGORIES:
            items1 = {i["id"]: i for i in r1.get(cat_key, [])}
            items2 = {i["id"]: i for i in r2.get(cat_key, [])}
            for cid in set(items1) & set(items2):
                s1 = is_satisfied(items1[cid])
                s2 = is_satisfied(items2[cid])
                if s1:
                    overall_sat += 1;  cat_sat[cat_key] += 1
                    if not s2:
                        overall_reg += 1; cat_reg[cat_key] += 1
                else:
                    overall_unsat += 1; cat_unsat[cat_key] += 1
                    if s2:
                        overall_imp += 1; cat_imp[cat_key] += 1

    def rate(n, d): return round(n / d * 100, 2) if d > 0 else None

    per_category = {}
    for cat_key, short in CATEGORIES:
        per_category[short] = {
            "incorp_rate": rate(cat_imp[cat_key], cat_unsat[cat_key]),
            "regression_rate": rate(cat_reg[cat_key], cat_sat[cat_key]),
            "improved": cat_imp[cat_key], "unsat": cat_unsat[cat_key],
            "regressed": cat_reg[cat_key], "sat": cat_sat[cat_key],
        }

    return {
        "n_tasks": len(shared),
        "overall": {
            "incorp_rate": rate(overall_imp, overall_unsat),
            "regression_rate": rate(overall_reg, overall_sat),
            "improved": overall_imp, "unsat": overall_unsat,
            "regressed": overall_reg, "sat": overall_sat,
        },
        "per_category": per_category,
    }


def compute_ir_by_domain(v1_evals: dict, v2_evals: dict, domains: dict) -> dict[str, dict]:
    shared = sorted(set(v1_evals) & set(v2_evals))
    by_domain: dict[str, list] = defaultdict(list)
    for task_id in shared:
        by_domain[domains.get(task_id, "Unknown")].append(task_id)

    result = {}
    for domain, tasks in sorted(by_domain.items()):
        sub_v1 = {t: v1_evals[t] for t in tasks}
        sub_v2 = {t: v2_evals[t] for t in tasks}
        result[domain] = compute_ir(sub_v1, sub_v2)
    return result


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_score_comparison(t1: dict, sr: dict, fb: dict) -> None:
    """Three-column score table: T1 | self-reflect | feedback."""
    W = 90
    print(f"\n  {'─'*W}")
    print(f"  {'Metric':<28} {'T1 NS':>8} {'T1 PR':>8} {'SR NS':>8} {'SR PR':>8} {'SR Δ':>8} {'FB NS':>8} {'FB PR':>8} {'FB Δ':>8}")
    print(f"  {'─'*W}")

    def _row(label, t1s, srs, fbs):
        ns1, pr1 = fmt(t1s["avg_ns"]), fmt(t1s["avg_pr"])
        ns_sr, pr_sr = fmt(srs["avg_ns"]), fmt(srs["avg_pr"])
        ns_fb, pr_fb = fmt(fbs["avg_ns"]), fmt(fbs["avg_pr"])
        d_sr = delta_str(round(srs["avg_ns"] - t1s["avg_ns"], 2) if t1s["avg_ns"] is not None and srs["avg_ns"] is not None else None)
        d_fb = delta_str(round(fbs["avg_ns"] - t1s["avg_ns"], 2) if t1s["avg_ns"] is not None and fbs["avg_ns"] is not None else None)
        print(f"  {label:<28} {ns1:>8} {pr1:>8} {ns_sr:>8} {pr_sr:>8} {d_sr:>8} {ns_fb:>8} {pr_fb:>8} {d_fb:>8}")

    _row("Overall", t1, sr, fb)
    print()
    for _, short in CATEGORIES:
        t1c  = {"avg_ns": t1["per_category"][short]["avg_ns"],  "avg_pr": t1["per_category"][short]["avg_pr"]}
        src  = {"avg_ns": sr["per_category"][short]["avg_ns"],  "avg_pr": sr["per_category"][short]["avg_pr"]}
        fbc  = {"avg_ns": fb["per_category"][short]["avg_ns"],  "avg_pr": fb["per_category"][short]["avg_pr"]}
        _row(short, t1c, src, fbc)


def print_ir_comparison(ir_sr: dict, ir_fb: dict) -> None:
    """Side-by-side incorporation / regression for both methods."""
    print(f"\n  Incorporation / Regression Rates  (n={ir_sr['n_tasks']} paired tasks)")
    W = 80
    print(f"  {'─'*W}")
    print(f"  {'Metric':<28} {'SR Incorp':>12} {'SR Regress':>12} {'FB Incorp':>12} {'FB Regress':>12}")
    print(f"  {'─'*W}")

    def _ir_row(label, sr_data, fb_data):
        print(f"  {label:<28} "
              f"{fmt(sr_data['incorp_rate']):>12} {fmt(sr_data['regression_rate']):>12} "
              f"{fmt(fb_data['incorp_rate']):>12} {fmt(fb_data['regression_rate']):>12}")
        if "improved" in sr_data:
            print(f"  {'':28} "
                  f"  ({sr_data['improved']}/{sr_data['unsat']} u→s) ({sr_data['regressed']}/{sr_data['sat']} s→u)"
                  f"   ({fb_data['improved']}/{fb_data['unsat']} u→s) ({fb_data['regressed']}/{fb_data['sat']} s→u)")

    _ir_row("Overall", ir_sr["overall"], ir_fb["overall"])
    print()
    for _, short in CATEGORIES:
        _ir_row(short, ir_sr["per_category"][short], ir_fb["per_category"][short])


def print_domain_score_comparison(t1_dom: dict, sr_dom: dict, fb_dom: dict) -> None:
    all_domains = sorted(set(t1_dom) | set(sr_dom) | set(fb_dom))
    print(f"\n  Domain breakdown — Normalized Score & Pass Rate")
    W = 96
    print(f"  {'─'*W}")
    print(f"  {'Domain':<32} {'n':>4} {'T1 NS':>8} {'T1 PR':>8} {'SR NS':>8} {'SR PR':>8} {'FB NS':>8} {'FB PR':>8}")
    print(f"  {'─'*W}")
    for domain in all_domains:
        d1 = t1_dom.get(domain, {})
        ds = sr_dom.get(domain, {})
        df = fb_dom.get(domain, {})
        n = max(d1.get("n", 0), ds.get("n", 0), df.get("n", 0))
        print(f"  {domain:<32} {n:>4} "
              f"{fmt(d1.get('avg_ns')):>8} {fmt(d1.get('avg_pr')):>8} "
              f"{fmt(ds.get('avg_ns')):>8} {fmt(ds.get('avg_pr')):>8} "
              f"{fmt(df.get('avg_ns')):>8} {fmt(df.get('avg_pr')):>8}")


def print_domain_ir_comparison(sr_dom_ir: dict, fb_dom_ir: dict) -> None:
    all_domains = sorted(set(sr_dom_ir) | set(fb_dom_ir))
    print(f"\n  Domain breakdown — Incorporation / Regression")
    W = 80
    print(f"  {'─'*W}")
    print(f"  {'Domain':<32} {'n':>4} {'SR Incorp':>12} {'SR Regress':>12} {'FB Incorp':>12} {'FB Regress':>12}")
    print(f"  {'─'*W}")
    for domain in all_domains:
        ds = sr_dom_ir.get(domain, {})
        df = fb_dom_ir.get(domain, {})
        n = max(ds.get("n_tasks", 0), df.get("n_tasks", 0))
        ds_ov = ds.get("overall", {})
        df_ov = df.get("overall", {})
        print(f"  {domain:<32} {n:>4} "
              f"{fmt(ds_ov.get('incorp_rate')):>12} {fmt(ds_ov.get('regression_rate')):>12} "
              f"{fmt(df_ov.get('incorp_rate')):>12} {fmt(df_ov.get('regression_rate')):>12}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare self-reflection baseline vs feedback method for v2 reports."
    )
    parser.add_argument(
        "--model", default="openai:gpt-4.1-mini",
        help="Model slug to compare (default: openai:gpt-4.1-mini)"
    )
    parser.add_argument(
        "--deepseekv4flash", action="store_true",
        help="Analyse DeepSeek v4 Flash results instead of GPT models."
    )
    args = parser.parse_args()

    slug = "deepseekv4flash" if args.deepseekv4flash else _model_slug(args.model)
    tasks_dir = ABLATIONS_DIR / "tasks"
    domains = load_task_domains(tasks_dir)

    v1_dir = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}"
    fb_dir = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}"
    sr_dir = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}_self_reflect"

    for d, label in [(v1_dir, "v1 evals"), (sr_dir, "self-reflect v2 evals")]:
        if not d.exists():
            raise FileNotFoundError(f"{label} directory not found: {d}")

    v1_evals = load_evals(v1_dir, "1")
    sr_evals = load_evals(sr_dir, "2")
    fb_evals = load_evals(fb_dir, "2")

    # Only tasks present in all three sets
    shared = sorted(set(v1_evals) & set(sr_evals) & set(fb_evals))
    v1_evals = {t: v1_evals[t] for t in shared}
    sr_evals = {t: sr_evals[t] for t in shared}
    fb_evals = {t: fb_evals[t] for t in shared}

    print(f"\n{'='*70}")
    print(f"  SELF-REFLECTION vs FEEDBACK COMPARISON — {slug.upper()}")
    print(f"  Tasks compared: {len(shared)}")
    print(f"  SR = self-reflection  |  FB = feedback method")
    print(f"{'='*70}")

    # --- score stats ---
    t1_stats = compute_score_stats(v1_evals)
    sr_stats = compute_score_stats(sr_evals)
    fb_stats = compute_score_stats(fb_evals)

    print(f"\n  Normalized Score & Pass Rate  (Δ = method v2 − T1)")
    print(f"  SR Δ = self-reflect delta from T1   |   FB Δ = feedback delta from T1")
    print_score_comparison(t1_stats, sr_stats, fb_stats)

    # --- IR rates ---
    ir_sr = compute_ir(v1_evals, sr_evals)
    ir_fb = compute_ir(v1_evals, fb_evals)
    print_ir_comparison(ir_sr, ir_fb)

    # --- domain breakdown ---
    t1_dom = compute_score_stats_by_domain(v1_evals, domains)
    sr_dom = compute_score_stats_by_domain(sr_evals, domains)
    fb_dom = compute_score_stats_by_domain(fb_evals, domains)
    print_domain_score_comparison(t1_dom, sr_dom, fb_dom)

    sr_dom_ir = compute_ir_by_domain(v1_evals, sr_evals, domains)
    fb_dom_ir = compute_ir_by_domain(v1_evals, fb_evals, domains)
    print_domain_ir_comparison(sr_dom_ir, fb_dom_ir)

    # --- save JSON ---
    analysis_dir = ABLATIONS_DIR / "analysis" / f"analysis_{slug}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "n_tasks": len(shared),
        "score_stats": {
            "t1": t1_stats, "self_reflect": sr_stats, "feedback": fb_stats
        },
        "ir_rates": {
            "self_reflect": ir_sr, "feedback": ir_fb
        },
        "domain_score_stats": {
            domain: {
                "t1":           t1_dom.get(domain, {}),
                "self_reflect": sr_dom.get(domain, {}),
                "feedback":     fb_dom.get(domain, {}),
            }
            for domain in sorted(set(t1_dom) | set(sr_dom) | set(fb_dom))
        },
        "domain_ir_rates": {
            domain: {
                "self_reflect": sr_dom_ir.get(domain, {}),
                "feedback":     fb_dom_ir.get(domain, {}),
            }
            for domain in sorted(set(sr_dom_ir) | set(fb_dom_ir))
        },
    }
    out_path = analysis_dir / "self_reflect_comparison.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  [Saved] {out_path}\n")


if __name__ == "__main__":
    main()
