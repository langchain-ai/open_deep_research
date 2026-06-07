#!/usr/bin/env python3
"""
Analyse citation overlap across ablation version pairs for a given model.

Runs three comparisons over all available tasks:
  - v1 vs v2            (base ablation versions)
  - v1 vs self-reflect  (v1 baseline vs self-reflection v2)
  - v2 vs v3            (incremental improvement)

Produces: ablations/analysis/analyse_citations/<model>.md

Usage:
    python ablations/scripts/analyse_citations.py gpt4.1
    python ablations/scripts/analyse_citations.py gpt4.1mini
    python ablations/scripts/analyse_citations.py deepseekv4flash
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import date

SCRIPT_DIR = Path(__file__).parent
ABLATIONS_DIR = SCRIPT_DIR.parent
OUT_DIR = ABLATIONS_DIR / "analysis" / "analyse_citations"
TITLE_THRESHOLD = 0.8

sys.path.insert(0, str(SCRIPT_DIR))
from compare_reports import extract_citations, match_citations, compute_ngram_overlap


@dataclass
class TaskResult:
    task: str
    cites_a: int
    cites_b: int
    url_matches: int
    title_matches: int
    total_matches: int
    pct_a_in_b: float
    pct_b_from_a: float
    only_a: int
    only_b: int


def compare_pair(path_a: Path, path_b: Path, task_id: str) -> Optional[TaskResult]:
    try:
        text_a = path_a.read_text(encoding="utf-8")
        text_b = path_b.read_text(encoding="utf-8")
    except OSError as e:
        print(f"  Warning: cannot read files for {task_id}: {e}", file=sys.stderr)
        return None

    cites_a = extract_citations(text_a)
    cites_b = extract_citations(text_b)

    if not cites_a or not cites_b:
        print(f"  Warning: skipping {task_id} — zero citations in {'A' if not cites_a else 'B'}", file=sys.stderr)
        return None

    url_m, title_m, unmatched_a, unmatched_b = match_citations(
        cites_a, cites_b, TITLE_THRESHOLD
    )
    total = len(url_m) + len(title_m)
    pct_a = total / len(cites_a) * 100 if cites_a else 0.0
    pct_b = total / len(cites_b) * 100 if cites_b else 0.0

    return TaskResult(
        task=task_id,
        cites_a=len(cites_a),
        cites_b=len(cites_b),
        url_matches=len(url_m),
        title_matches=len(title_m),
        total_matches=total,
        pct_a_in_b=pct_a,
        pct_b_from_a=pct_b,
        only_a=len(unmatched_a),
        only_b=len(unmatched_b),
    )


def build_pairs(
    dir_a: Path,
    suffix_a: str,
    dir_b: Path,
    suffix_b: str,
) -> List[Tuple[Path, Path, str]]:
    """Find all task IDs where both files exist and return (path_a, path_b, task_id) triples."""
    pairs = []
    for path_a in sorted(dir_a.glob(f"task_*{suffix_a}.md")):
        task_id = path_a.stem[: -len(suffix_a)]  # strip e.g. "_v1"
        path_b = dir_b / f"{task_id}{suffix_b}.md"
        if path_b.exists():
            pairs.append((path_a, path_b, task_id))
    return pairs


def run_comparison(
    pairs: List[Tuple[Path, Path, str]],
) -> List[TaskResult]:
    results = []
    for path_a, path_b, task_id in pairs:
        r = compare_pair(path_a, path_b, task_id)
        if r is not None:
            results.append(r)
    return results


def avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def run_ngram_comparison(
    pairs: List[Tuple[Path, Path, str]], n: int = 5
) -> List[dict]:
    results = []
    for path_a, path_b, task_id in pairs:
        try:
            text_a = path_a.read_text(encoding="utf-8")
            text_b = path_b.read_text(encoding="utf-8")
        except OSError:
            continue
        ng = compute_ngram_overlap(text_a, text_b, n=n)
        ng["task_id"] = task_id
        results.append(ng)
    return results


def summarise_ngrams(results: List[dict]) -> dict:
    recalls    = [r["recall"]    for r in results if r["recall"]    is not None]
    precisions = [r["precision"] for r in results if r["precision"] is not None]
    f1s        = [r["f1"]        for r in results if r["f1"]        is not None]
    return {
        "n_tasks":        len(results),
        "avg_recall":     avg(recalls),
        "avg_precision":  avg(precisions),
        "avg_f1":         avg(f1s),
    }


def summarise(results: List[TaskResult]) -> dict:
    if not results:
        return {}
    return {
        "n_tasks": len(results),
        "avg_cites_a": avg([r.cites_a for r in results]),
        "avg_cites_b": avg([r.cites_b for r in results]),
        "avg_url_matches": avg([r.url_matches for r in results]),
        "avg_title_matches": avg([r.title_matches for r in results]),
        "avg_total_matches": avg([r.total_matches for r in results]),
        "avg_pct_a_in_b": avg([r.pct_a_in_b for r in results]),
        "avg_pct_b_from_a": avg([r.pct_b_from_a for r in results]),
        "avg_only_a": avg([r.only_a for r in results]),
        "avg_only_b": avg([r.only_b for r in results]),
    }


def render_summary_section(label: str, s: dict, pair_desc: Tuple[str, str]) -> str:
    label_a, label_b = pair_desc
    if not s:
        return f"### {label}\n\n_No results found._\n\n"

    lines = [
        f"### {label}",
        "",
        f"Tasks analysed: **{s['n_tasks']}**  ",
        f"Title-match threshold: {TITLE_THRESHOLD:.0%}",
        "",
        "| Metric | Average |",
        "|--------|---------|",
        f"| Citations in {label_a} | {s['avg_cites_a']:.1f} |",
        f"| Citations in {label_b} | {s['avg_cites_b']:.1f} |",
        f"| URL matches | {s['avg_url_matches']:.1f} |",
        f"| Title matches | {s['avg_title_matches']:.1f} |",
        f"| **Total matched** | **{s['avg_total_matches']:.1f}** |",
        f"| {label_a} retained in {label_b} (%) | {s['avg_pct_a_in_b']:.1f}% |",
        f"| {label_b} from {label_a} (%) | {s['avg_pct_b_from_a']:.1f}% |",
        f"| Only in {label_a} | {s['avg_only_a']:.1f} |",
        f"| Only in {label_b} | {s['avg_only_b']:.1f} |",
        "",
    ]
    return "\n".join(lines)


def render_per_task_table(label: str, results: List[TaskResult], pair_desc: Tuple[str, str]) -> str:
    label_a, label_b = pair_desc
    if not results:
        return ""
    header = (
        f"#### Per-task breakdown: {label}\n\n"
        f"| Task | Cites {label_a} | Cites {label_b} | URL | Title | Total | "
        f"{label_a}→{label_b} % | {label_b}←{label_a} % |\n"
        f"|------|----------|----------|-----|-------|-------|---------|---------|"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.task} | {r.cites_a} | {r.cites_b} | {r.url_matches} | "
            f"{r.title_matches} | {r.total_matches} | "
            f"{r.pct_a_in_b:.1f}% | {r.pct_b_from_a:.1f}% |"
        )
    return header + "\n" + "\n".join(rows) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python ablations/scripts/analyse_citations.py <model>")
        print("  model: gpt4.1 | gpt4.1mini | deepseekv4flash")
        sys.exit(1)

    model = sys.argv[1]

    base_dir = ABLATIONS_DIR / "reports" / f"reports_{model}"
    sr_dir = ABLATIONS_DIR / "reports" / f"reports_{model}_self_reflect"

    if not base_dir.exists():
        print(f"Error: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)
    if not sr_dir.exists():
        print(f"Warning: self-reflect directory not found: {sr_dir}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model}.md"

    # ── build pair lists ──────────────────────────────────────────────────────
    pairs_v1v2 = build_pairs(base_dir, "_v1", base_dir, "_v2")
    pairs_v1sr = build_pairs(base_dir, "_v1", sr_dir, "_v2") if sr_dir.exists() else []
    pairs_v2v3 = build_pairs(base_dir, "_v2", base_dir, "_v3")

    print(f"Model: {model}")
    print(f"  v1 vs v2:            {len(pairs_v1v2)} tasks")
    print(f"  v1 vs self-reflect:  {len(pairs_v1sr)} tasks")
    print(f"  v2 vs v3:            {len(pairs_v2v3)} tasks")

    # ── run comparisons ───────────────────────────────────────────────────────
    print("\nRunning v1 vs v2 ...")
    res_v1v2 = run_comparison(pairs_v1v2)

    print("Running v1 vs self-reflect ...")
    res_v1sr = run_comparison(pairs_v1sr)

    print("Running v2 vs v3 ...")
    res_v2v3 = run_comparison(pairs_v2v3)

    # ── n-gram overlap (v2 vs v3) ─────────────────────────────────────────────
    NGRAM_N = 7
    print(f"\nComputing {NGRAM_N}-gram overlap (v2 vs v3) ...")
    ng_v2v3 = run_ngram_comparison(pairs_v2v3, n=NGRAM_N)
    ng_sum   = summarise_ngrams(ng_v2v3)

    print(f"\n{'='*50}")
    print(f"  {NGRAM_N}-GRAM BODY OVERLAP — v2 vs v3  (n={ng_sum['n_tasks']} tasks)")
    print(f"{'='*50}")
    print(f"  Recall  (v2→v3):  {ng_sum['avg_recall']:.2f}%  "
          f"(avg fraction of v2 {NGRAM_N}-grams present in v3)")
    print(f"  Precision (v3←v2):{ng_sum['avg_precision']:.2f}%  "
          f"(avg fraction of v3 {NGRAM_N}-grams sourced from v2)")
    print(f"  F1:               {ng_sum['avg_f1']:.2f}%")
    print(f"{'='*50}\n")

    # ── summarise ─────────────────────────────────────────────────────────────
    sum_v1v2 = summarise(res_v1v2)
    sum_v1sr = summarise(res_v1sr)
    sum_v2v3 = summarise(res_v2v3)

    # ── render markdown ───────────────────────────────────────────────────────
    md = [
        f"# Citation Overlap Analysis — {model}",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Title-match threshold: {TITLE_THRESHOLD:.0%}  ",
        "",
        "Compares citation overlap between ablation version pairs, averaged over all tasks.",
        "",
        "---",
        "",
        "## Averaged Summaries",
        "",
        render_summary_section("v1 vs v2", sum_v1v2, ("v1", "v2")),
        render_summary_section("v1 vs self-reflect", sum_v1sr, ("v1", "self-reflect")),
        render_summary_section("v2 vs v3", sum_v2v3, ("v2", "v3")),
        "---",
        "",
        "## Per-task Breakdowns",
        "",
        render_per_task_table("v1 vs v2", res_v1v2, ("v1", "v2")),
        "",
        render_per_task_table("v1 vs self-reflect", res_v1sr, ("v1", "sr")),
        "",
        render_per_task_table("v2 vs v3", res_v2v3, ("v2", "v3")),
    ]

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
