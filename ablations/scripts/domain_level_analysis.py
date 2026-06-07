"""Domain-level analysis across Turn-1, Turn-2, Turn-3, and Self-Reflection.

For each model × domain reports:
  - Norm score and pass rate at T1, T2, T3, self-reflect (SR), feedback (FB=T2)
  - Incorporation rate, regression rate, and net criteria gain for:
      T1 → T2  (feedback)
      T2 → T3  (feedback)
      T1 → T3  (overall, feedback)
      T1 → SR  (self-reflection)
      T1 → FB  (= T1 → T2, same as feedback turn-2)

Net criteria gain = improved_count − regressed_count  (absolute)
Net criteria gain % = net / total_criteria × 100

Saves: ablations/analysis/analysis_<slug>/domain_level_analysis.json

Usage:
    python ablations/scripts/domain_level_analysis.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ABLATIONS_DIR = Path(__file__).parent.parent

MODELS_GPT = [
    ("GPT-4.1",      "gpt4.1"),
    ("GPT-4.1-mini", "gpt4.1mini"),
]

MODELS_DEEPSEEK = [
    ("DeepSeek v4 Flash", "deepseekv4flash"),
]

CATEGORIES = [
    ("factual-accuracy",              "FA"),
    ("breadth-and-depth-of-analysis", "BD"),
    ("presentation-quality",          "PQ"),
    ("citation-quality",              "CQ"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def rate(n: int, d: int) -> float | None:
    return round(n / d * 100, 2) if d > 0 else None


def fmt(v) -> str:
    return f"{v:>7.2f}%" if v is not None else "    N/A"


def delta_str(v: float | None) -> str:
    if v is None:
        return "   N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def is_satisfied(item: dict) -> bool:
    return (not item["is_negative"] and item["verdict"] == "MET") or \
           (item["is_negative"] and item["verdict"] == "UNMET")


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_evals(evals_dir: Path, turn: str) -> dict[str, dict]:
    result = {}
    for f in sorted(evals_dir.glob(f"*_v{turn}_eval.json")):
        data = json.loads(f.read_text())
        result[data["task_id"]] = data
    return result


def load_evals_with_fallback(evals_dir: Path, turn: str, fallback: str) -> dict[str, dict]:
    primary = load_evals(evals_dir, turn)
    fb_map  = load_evals(evals_dir, fallback)
    merged  = {}
    for tid in fb_map:
        merged[tid] = primary.get(tid, fb_map[tid])
    return merged


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
# score stats
# ---------------------------------------------------------------------------

def compute_score_stats(evals: dict[str, dict]) -> dict:
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
        per_category[short] = {"avg_ns": avg(cat_ns[cat_key]), "avg_pr": avg(cat_pr[cat_key])}

    return {"n": len(evals), "avg_ns": avg(ns_list), "avg_pr": avg(pr_list),
            "per_category": per_category}


def score_stats_by_domain(evals: dict, domains: dict) -> dict[str, dict]:
    by_domain: dict[str, dict] = defaultdict(dict)
    for tid, data in evals.items():
        by_domain[domains.get(tid, "Unknown")][tid] = data
    return {d: compute_score_stats(sub) for d, sub in sorted(by_domain.items())}


# ---------------------------------------------------------------------------
# incorporation / regression / net gain
# ---------------------------------------------------------------------------

def compute_ir(prev_evals: dict, next_evals: dict) -> dict:
    shared = sorted(set(prev_evals) & set(next_evals))

    ov_unsat = ov_sat = ov_imp = ov_reg = 0
    cat_unsat: dict[str, int] = defaultdict(int)
    cat_sat:   dict[str, int] = defaultdict(int)
    cat_imp:   dict[str, int] = defaultdict(int)
    cat_reg:   dict[str, int] = defaultdict(int)

    for tid in shared:
        r1 = prev_evals[tid].get("results", {})
        r2 = next_evals[tid].get("results", {})
        for cat_key, _ in CATEGORIES:
            items1 = {i["id"]: i for i in r1.get(cat_key, [])}
            items2 = {i["id"]: i for i in r2.get(cat_key, [])}
            for cid in set(items1) & set(items2):
                s1 = is_satisfied(items1[cid])
                s2 = is_satisfied(items2[cid])
                if s1:
                    ov_sat += 1; cat_sat[cat_key] += 1
                    if not s2:
                        ov_reg += 1; cat_reg[cat_key] += 1
                else:
                    ov_unsat += 1; cat_unsat[cat_key] += 1
                    if s2:
                        ov_imp += 1; cat_imp[cat_key] += 1

    total = ov_unsat + ov_sat
    net   = ov_imp - ov_reg

    per_category = {}
    for cat_key, short in CATEGORIES:
        tot_c = cat_unsat[cat_key] + cat_sat[cat_key]
        net_c = cat_imp[cat_key] - cat_reg[cat_key]
        per_category[short] = {
            "incorp_rate":    rate(cat_imp[cat_key], cat_unsat[cat_key]),
            "regression_rate": rate(cat_reg[cat_key], cat_sat[cat_key]),
            "net_gain":       net_c,
            "net_gain_pct":   rate(net_c, tot_c) if tot_c else None,
            "improved": cat_imp[cat_key], "unsat": cat_unsat[cat_key],
            "regressed": cat_reg[cat_key], "sat": cat_sat[cat_key],
        }

    return {
        "n_tasks": len(shared),
        "overall": {
            "incorp_rate":    rate(ov_imp, ov_unsat),
            "regression_rate": rate(ov_reg, ov_sat),
            "net_gain":       net,
            "net_gain_pct":   rate(net, total) if total else None,
            "improved": ov_imp, "unsat": ov_unsat,
            "regressed": ov_reg, "sat": ov_sat,
            "total_criteria": total,
        },
        "per_category": per_category,
    }


def ir_by_domain(prev_evals: dict, next_evals: dict, domains: dict) -> dict[str, dict]:
    shared = sorted(set(prev_evals) & set(next_evals))
    by_domain: dict[str, list] = defaultdict(list)
    for tid in shared:
        by_domain[domains.get(tid, "Unknown")].append(tid)
    return {
        domain: compute_ir(
            {t: prev_evals[t] for t in tasks},
            {t: next_evals[t] for t in tasks},
        )
        for domain, tasks in sorted(by_domain.items())
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_score_table(label: str, domain_scores: dict[str, dict[str, dict]]) -> None:
    """domain_scores[domain][turn_key] = score_stats dict (avg_ns, avg_pr)."""
    turns = list(next(iter(domain_scores.values())).keys())
    header_parts = [f"{'Domain':<32}", f"{'n':>4}"]
    for t in turns:
        header_parts += [f"{t+' NS':>9}", f"{t+' PR':>9}"]
    print("  " + "  ".join(header_parts))
    print("  " + "─" * (32 + 4 + len(turns) * 20 + 4))
    for domain, t_data in sorted(domain_scores.items()):
        n = max(v.get("n", 0) for v in t_data.values())
        row = f"  {domain:<32} {n:>4}"
        for t in turns:
            s = t_data.get(t, {})
            row += f"  {fmt(s.get('avg_ns')):>9}{fmt(s.get('avg_pr')):>9}"
        print(row)


def print_ir_table(label: str, domain_ir: dict[str, dict[str, dict]],
                   transitions: list[str]) -> None:
    """domain_ir[domain][transition] = ir dict."""
    print(f"\n  {label}")
    hdr = f"  {'Domain':<32} {'n':>4}"
    for t in transitions:
        hdr += f"  {t+' Incorp':>12} {t+' Regress':>12} {t+' NetGain':>10}"
    print(hdr)
    print("  " + "─" * (32 + 4 + len(transitions) * 38 + 4))
    all_domains = sorted(set().union(*[set(domain_ir[tr].keys()) for tr in transitions
                                       if tr in domain_ir]))
    for domain in all_domains:
        n = 0
        row = f"  {domain:<32}"
        for t in transitions:
            ir = domain_ir.get(t, {}).get(domain, {})
            ov = ir.get("overall", {})
            n = max(n, ir.get("n_tasks", 0))
            net = ov.get("net_gain")
            net_str = (f"+{net}" if net is not None and net >= 0 else str(net)) if net is not None else "N/A"
            row += (f"  {fmt(ov.get('incorp_rate')):>12}"
                    f" {fmt(ov.get('regression_rate')):>12}"
                    f" {net_str:>10}")
        print(f"{row[:36]}{n:>4}{row[36:]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-level analysis across turns.")
    parser.add_argument("--deepseekv4flash", action="store_true",
                        help="Analyse DeepSeek v4 Flash results instead of GPT models.")
    args = parser.parse_args()

    models = MODELS_DEEPSEEK if args.deepseekv4flash else MODELS_GPT

    tasks_dir = ABLATIONS_DIR / "tasks"
    domains   = load_task_domains(tasks_dir)

    for model_name, slug in models:
        fb_dir = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}"
        sr_dir = ABLATIONS_DIR / "evaluations" / f"evaluations_{slug}_self_reflect"

        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*70}")

        # --- load eval sets ---
        v1 = load_evals(fb_dir, "1")
        v2 = load_evals(fb_dir, "2")
        v3 = load_evals_with_fallback(fb_dir, "3", fallback="2")
        sr = load_evals(sr_dir, "2") if sr_dir.exists() else {}

        # restrict to tasks present everywhere
        shared_base = sorted(set(v1) & set(v2))
        shared_all  = sorted(set(v1) & set(v2) & set(v3))
        shared_sr   = sorted(set(v1) & set(sr)) if sr else []

        # --- score stats by domain ---
        t1_dom = score_stats_by_domain({t: v1[t] for t in shared_all}, domains)
        t2_dom = score_stats_by_domain({t: v2[t] for t in shared_all}, domains)
        t3_dom = score_stats_by_domain({t: v3[t] for t in shared_all}, domains)
        sr_dom = score_stats_by_domain({t: sr[t] for t in shared_sr}, domains) if shared_sr else {}

        all_domains = sorted(set(t1_dom) | set(t2_dom) | set(t3_dom) | set(sr_dom))

        # build combined score dict
        score_by_domain: dict[str, dict] = {}
        for d in all_domains:
            score_by_domain[d] = {
                "T1": t1_dom.get(d, {}),
                "T2": t2_dom.get(d, {}),
                "T3": t3_dom.get(d, {}),
            }
            if sr_dom:
                score_by_domain[d]["SR"] = sr_dom.get(d, {})

        print(f"\n  Normalized Score & Pass Rate — by Domain")
        print_score_table("Scores", score_by_domain)

        # --- IR by domain ---
        ir_t1_t2 = ir_by_domain(
            {t: v1[t] for t in shared_base},
            {t: v2[t] for t in shared_base},
            domains,
        )
        ir_t2_t3 = ir_by_domain(
            {t: v2[t] for t in shared_all},
            {t: v3[t] for t in shared_all},
            domains,
        )
        ir_t1_sr = ir_by_domain(
            {t: v1[t] for t in shared_sr},
            {t: sr[t] for t in shared_sr},
            domains,
        ) if shared_sr else {}

        transitions = ["T1→T2", "T2→T3"]
        domain_ir_map = {
            "T1→T2": ir_t1_t2,
            "T2→T3": ir_t2_t3,
        }
        if ir_t1_sr:
            transitions.append("T1→SR")
            domain_ir_map["T1→SR"] = ir_t1_sr

        print_ir_table("Incorporation / Regression / Net Criteria Gain — by Domain",
                       domain_ir_map, transitions)

        # --- save JSON ---
        analysis_dir = ABLATIONS_DIR / "analysis" / f"analysis_{slug}"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        def score_overall(s: dict) -> dict:
            return {"n": s.get("n"), "avg_ns": s.get("avg_ns"), "avg_pr": s.get("avg_pr")}

        def ir_overall(ir: dict) -> dict:
            ov = ir.get("overall", {})
            return {
                "n_tasks":         ir.get("n_tasks"),
                "incorp_rate":     ov.get("incorp_rate"),
                "regression_rate": ov.get("regression_rate"),
                "net_gain":        ov.get("net_gain"),
                "improved":        ov.get("improved"),
                "unsat":           ov.get("unsat"),
                "regressed":       ov.get("regressed"),
                "sat":             ov.get("sat"),
                "total_criteria":  ov.get("total_criteria"),
            }

        out = {
            "model": model_name,
            "domains": {}
        }
        for d in all_domains:
            entry: dict = {
                "score": {
                    "T1": score_overall(t1_dom.get(d, {})),
                    "T2": score_overall(t2_dom.get(d, {})),
                    "T3": score_overall(t3_dom.get(d, {})),
                },
                "ir": {
                    "T1_to_T2": ir_overall(ir_t1_t2.get(d, {})),
                    "T2_to_T3": ir_overall(ir_t2_t3.get(d, {})),
                }
            }
            if sr_dom:
                entry["score"]["SR"] = score_overall(sr_dom.get(d, {}))
            if ir_t1_sr:
                entry["ir"]["T1_to_SR"] = ir_overall(ir_t1_sr.get(d, {}))
            out["domains"][d] = entry

        out_path = analysis_dir / "domain_level_analysis.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  [Saved] {out_path}")

    print()


if __name__ == "__main__":
    main()
