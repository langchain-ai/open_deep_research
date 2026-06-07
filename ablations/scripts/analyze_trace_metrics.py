"""Analyze ablations/trace_metrics/trace_metrics_merged.json (GPT models) or
ablations/trace_metrics/trace_metrics_deepseekv4flash.json (DeepSeek v4 Flash).

GPT mode (default) — generates 6 output files in ablations/analysis/analysis_trace_level/:
  gpt4.1_turns.md              — GPT-4.1  v1/v2/v3 overall trace metrics + report characteristics
  gpt4.1_self_reflect.md       — GPT-4.1  T1 vs self-reflect overall metrics
  gpt4.1mini_turns.md          — GPT-4.1-mini v1/v2/v3 overall trace metrics + report characteristics
  gpt4.1mini_self_reflect.md   — GPT-4.1-mini T1 vs self-reflect overall metrics
  gpt4.1_domain.md             — GPT-4.1  domain-level breakdown (turns + self-reflect)
  gpt4.1mini_domain.md         — GPT-4.1-mini domain-level breakdown (turns + self-reflect)

DeepSeek mode (--deepseekv4flash) — generates 3 output files in ablations/analysis/analysis_deepseekv4flash/:
  deepseekv4flash_turns.md        — DeepSeek v4 Flash v1/v2/v3 overall trace metrics + report characteristics
  deepseekv4flash_self_reflect.md — DeepSeek v4 Flash T1 vs self-reflect overall metrics
  deepseekv4flash_domain.md       — DeepSeek v4 Flash domain-level breakdown (turns + self-reflect)

Usage:
    uv run python ablations/scripts/analyze_trace_metrics.py
    uv run python ablations/scripts/analyze_trace_metrics.py --input ablations/trace_metrics/trace_metrics_merged.json
    uv run python ablations/scripts/analyze_trace_metrics.py --deepseekv4flash
    uv run python ablations/scripts/analyze_trace_metrics.py --deepseekv4flash --input ablations/trace_metrics/trace_metrics_deepseekv4flash.json
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

_ABLATIONS_DIR        = Path(__file__).parent.parent
TASKS_DIR             = _ABLATIONS_DIR / "tasks"
ANALYSIS_DIR          = _ABLATIONS_DIR / "analysis" / "analysis_trace_level"
ANALYSIS_DEEPSEEK_DIR = _ABLATIONS_DIR / "analysis" / "analysis_deepseekv4flash"

REPORT_DIRS = {
    "gpt41":                _ABLATIONS_DIR / "reports" / "reports_gpt4.1",
    "gpt41_self_reflect":   _ABLATIONS_DIR / "reports" / "reports_gpt4.1_self_reflect",
    "mini":                 _ABLATIONS_DIR / "reports" / "reports_gpt4.1mini",
    "mini_self_reflect":    _ABLATIONS_DIR / "reports" / "reports_gpt4.1mini_self_reflect",
    "deepseek":             _ABLATIONS_DIR / "reports" / "reports_deepseekv4flash",
    "deepseek_self_reflect":_ABLATIONS_DIR / "reports" / "reports_deepseekv4flash_self_reflect",
}

MODEL_CONFIGS_GPT = {
    "gpt4.1": {
        "turn_project_labels": ["gpt41_turn1", "gpt41_turn2", "gpt41_turn3"],
        "sr_project_labels":   ["gpt41_turn1", "gpt41_self_reflect"],
        "report_key":          "gpt41",
        "report_sr_key":       "gpt41_self_reflect",
        "turns_output":        ANALYSIS_DIR / "gpt4.1_turns.md",
        "sr_output":           ANALYSIS_DIR / "gpt4.1_self_reflect.md",
        "domain_output":       ANALYSIS_DIR / "gpt4.1_domain.md",
    },
    "gpt4.1mini": {
        "turn_project_labels": ["mini_turn1", "mini_turn2", "mini_turn3"],
        "sr_project_labels":   ["mini_turn1", "mini_self_reflect"],
        "report_key":          "mini",
        "report_sr_key":       "mini_self_reflect",
        "turns_output":        ANALYSIS_DIR / "gpt4.1mini_turns.md",
        "sr_output":           ANALYSIS_DIR / "gpt4.1mini_self_reflect.md",
        "domain_output":       ANALYSIS_DIR / "gpt4.1mini_domain.md",
    },
}

MODEL_CONFIGS_DEEPSEEK = {
    "deepseekv4flash": {
        "turn_project_labels": ["deepseek_turn1", "deepseek_turn2", "deepseek_turn3"],
        "sr_project_labels":   ["deepseek_turn1", "deepseek_self_reflect"],
        "report_key":          "deepseek",
        "report_sr_key":       "deepseek_self_reflect",
        "turns_output":        ANALYSIS_DEEPSEEK_DIR / "deepseekv4flash_turns.md",
        "sr_output":           ANALYSIS_DEEPSEEK_DIR / "deepseekv4flash_self_reflect.md",
        "domain_output":       ANALYSIS_DEEPSEEK_DIR / "deepseekv4flash_domain.md",
    },
}

# backward-compat alias used by existing code paths
MODEL_CONFIGS = MODEL_CONFIGS_GPT


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


def task_id_from_run_name(run_name: str) -> str:
    s = re.sub(r'_turn\d+$', '', run_name)
    s = re.sub(r'_self_reflect$', '', s)
    return s


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "total": None, "n": 0}
    return {
        "mean":   round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "total":  round(sum(values), 4),
        "n":      len(values),
    }


def _agg_int(values: list) -> dict:
    if not values:
        return {"mean": None, "median": None, "max": None, "total": None, "n": 0}
    return {
        "mean":   round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "max":    max(values),
        "total":  sum(values),
        "n":      len(values),
    }


def _mean(lst: list) -> float | None:
    return round(statistics.mean(lst), 2) if lst else None


# ---------------------------------------------------------------------------
# Reporter — prints to terminal and accumulates markdown
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self):
        self._lines: list[str] = []

    def _emit(self, text: str = "") -> None:
        print(text)
        self._lines.append(text)

    def section(self, title: str) -> None:
        self._emit(f"\n## {title}")

    def subsection(self, title: str) -> None:
        self._emit(f"\n### {title}")

    def line(self, text: str = "") -> None:
        self._emit(text)

    def table_row(self, cells: list, widths: list, aligns: list | None = None) -> None:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            align = aligns[i] if aligns else "l"
            s = str(cell)
            parts.append(s.rjust(w) if align == "r" else s.ljust(w))
        self._emit("| " + " | ".join(parts) + " |")

    def table_sep(self, widths: list, aligns: list | None = None) -> None:
        parts = []
        for i, w in enumerate(widths):
            align = aligns[i] if aligns else "l"
            parts.append(("-" * (w - 1) + ":") if align == "r" else (":" + "-" * (w - 1)))
        self._emit("| " + " | ".join(parts) + " |")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines), encoding="utf-8")
        print(f"\n[Saved] Report written to {path}")


# ---------------------------------------------------------------------------
# Overall tables
# ---------------------------------------------------------------------------

def print_overall(r: Reporter, projects: dict) -> None:
    r.section("Overall — Token / Cost / Latency")
    r.line("> mean per task; Total $ is summed across all tasks in the project")
    r.line()
    hdrs = ["Label", "n", "InTok (mean)", "OutTok (mean)", "Total $", "Avg Lat (s)"]
    ws   = [22, 4, 14, 14, 10, 12]
    al   = ["l","r","r","r","r","r"]
    r.table_row(hdrs, ws, al)
    r.table_sep(ws, al)
    for label, data in projects.items():
        if data.get("skipped"):
            continue
        agg = data["aggregates"]
        r.table_row([
            label,
            data["n_runs"],
            f"{agg['input_tokens']['mean'] or 0:,.0f}",
            f"{agg['output_tokens']['mean'] or 0:,.0f}",
            f"${agg['cost_usd']['total'] or 0:.4f}",
            f"{agg['latency_seconds']['mean'] or 0:.1f}s",
        ], ws, al)

    r.section("Overall — Count Metrics")
    r.line("> format: mean / median / max per task")
    r.line()
    hdrs = ["Label", "Researchers", "React iters", "Searches", "URLs"]
    ws   = [22, 16, 14, 14, 14]
    al   = ["l","r","r","r","r"]
    r.table_row(hdrs, ws, al)
    r.table_sep(ws, al)
    for label, data in projects.items():
        if data.get("skipped"):
            continue
        agg = data["aggregates"]
        def _f(k):
            a = agg[k]
            if a["mean"] is None:
                return "—"
            return f"{a['mean']:.1f} / {a['median']:.1f} / {a['max']}"
        r.table_row([label, _f("researcher_instances"), _f("researcher_react_iters"),
                     _f("search_calls"), _f("url_count")], ws, al)

    r.section("Overall — Phase Cost Breakdown")
    r.line("> mean $ per task | % of project total cost")
    r.line()
    all_phases = sorted({
        p for d in projects.values()
        if not d.get("skipped")
        for p in d.get("phase_aggregates", {})
    })
    hdrs = ["Label"] + all_phases
    ws   = [22] + [24] * len(all_phases)
    al   = ["l"] + ["r"] * len(all_phases)
    r.table_row(hdrs, ws, al)
    r.table_sep(ws, al)
    for label, data in projects.items():
        if data.get("skipped"):
            continue
        pa = data.get("phase_aggregates", {})
        cells = [label]
        for phase in all_phases:
            if phase in pa:
                cells.append(f"${pa[phase]['mean_cost_usd']:.4f} ({pa[phase]['pct_of_total_cost']}%)")
            else:
                cells.append("—")
        r.table_row(cells, ws, al)


def print_report_chars_turns(r: Reporter, report_analysis: dict, report_key: str) -> None:
    rdata = report_analysis.get(report_key, {})
    if "error" in rdata:
        r.line(f"> ERROR loading report characteristics: {rdata['error']}")
        return

    r.section("Report Characteristics")
    r.line("> mean per task across all tasks with that version")
    r.line()
    hdrs = ["Metric", "v1 mean", "v2 mean", "Δ v1→v2", "v3 mean", "Δ v2→v3"]
    ws   = [18, 10, 10, 10, 10, 10]
    al   = ["l","r","r","r","r","r"]
    r.table_row(hdrs, ws, al)
    r.table_sep(ws, al)

    agg = rdata["aggregates"]
    da  = rdata.get("delta_aggregates", {})

    for display_name, mean_key, delta_key in [
        ("word_count",     "word_count_mean",     "word_count_delta_mean"),
        ("citation_count", "citation_count_mean", "citation_count_delta_mean"),
        ("section_count",  "section_count_mean",  "section_count_delta_mean"),
    ]:
        v1m = agg.get("v1", {}).get(mean_key) or 0
        v2m = agg.get("v2", {}).get(mean_key) or 0
        v3m = agg.get("v3", {}).get(mean_key) or 0
        d12 = da.get("v1_to_v2", {}).get(delta_key) or 0
        d23 = da.get("v2_to_v3", {}).get(delta_key) or 0
        r.table_row([
            display_name,
            f"{v1m:.1f}", f"{v2m:.1f}", f"{d12:+.1f}",
            f"{v3m:.1f}", f"{d23:+.1f}",
        ], ws, al)


def print_report_chars_sr(
    r: Reporter,
    report_analysis: dict,
    report_key: str,
    report_sr_key: str,
) -> None:
    base_data = report_analysis.get(report_key, {})
    sr_data   = report_analysis.get(report_sr_key, {})

    if "error" in base_data or "error" in sr_data:
        r.line("> ERROR loading report characteristics")
        return

    r.section("Report Characteristics — T1 vs Self-Reflect v2")
    r.line()
    hdrs = ["Metric", "T1 (v1)", "Self-Reflect v2", "Δ (SR − T1)"]
    ws   = [18, 12, 16, 14]
    al   = ["l","r","r","r"]
    r.table_row(hdrs, ws, al)
    r.table_sep(ws, al)

    t1_v1 = base_data["aggregates"].get("v1", {})
    sr_v2 = sr_data["aggregates"].get("v2", {})

    for display_name, mean_key in [
        ("word_count",     "word_count_mean"),
        ("citation_count", "citation_count_mean"),
        ("section_count",  "section_count_mean"),
    ]:
        t1_val = t1_v1.get(mean_key) or 0
        sr_val = sr_v2.get(mean_key) or 0
        r.table_row([display_name, f"{t1_val:.1f}", f"{sr_val:.1f}", f"{sr_val - t1_val:+.1f}"], ws, al)


# ---------------------------------------------------------------------------
# Domain-level analysis
# ---------------------------------------------------------------------------

def build_domain_data(projects: dict, task_domains: dict) -> dict[str, dict[str, list]]:
    domain_data: dict[str, dict[str, list]] = {}
    for label, data in projects.items():
        if data.get("skipped"):
            continue
        domain_data[label] = defaultdict(list)
        for run_name, task in data["per_task"].items():
            tid = task_id_from_run_name(run_name)
            domain = task_domains.get(tid, "Unknown")
            domain_data[label][domain].append(task)
    return domain_data


def _domain_agg(tasks: list[dict]) -> dict:
    def _vals(key):
        return [t[key] for t in tasks if t.get(key) is not None]
    return {
        "n":                    len(tasks),
        "input_tokens":         _agg(_vals("input_tokens")),
        "output_tokens":        _agg(_vals("output_tokens")),
        "cost_usd":             _agg(_vals("cost_usd")),
        "latency_seconds":      _agg(_vals("latency_seconds")),
        "researcher_instances": _agg_int(_vals("researcher_instances")),
        "search_calls":         _agg_int(_vals("search_calls")),
        "url_count":            _agg_int(_vals("url_count")),
    }


def print_domain_analysis(r: Reporter, projects: dict, task_domains: dict, section_prefix: str = "") -> None:
    domain_data = build_domain_data(projects, task_domains)
    all_domains = sorted({d for ld in domain_data.values() for d in ld})
    labels      = list(domain_data.keys())
    w_domain    = 30

    prefix = f"{section_prefix} — " if section_prefix else ""

    # --- Tokens ---
    r.subsection(f"{prefix}Input / Output Tokens (mean per task)")
    r.line()
    col_hdrs = []
    for lbl in labels:
        col_hdrs += [f"InTok {lbl}", f"OutTok {lbl}"]
    ws = [w_domain] + [18] * len(col_hdrs)
    al = ["l"] + ["r"] * len(col_hdrs)
    r.table_row(["Domain"] + col_hdrs, ws, al)
    r.table_sep(ws, al)
    for domain in all_domains:
        cells = [domain]
        for label in labels:
            tasks = domain_data[label].get(domain, [])
            if not tasks:
                cells += ["—", "—"]
                continue
            agg   = _domain_agg(tasks)
            in_m  = agg["input_tokens"]["mean"]
            out_m = agg["output_tokens"]["mean"]
            cells.append(f"{in_m:,.0f} (n={agg['n']})" if in_m is not None else "—")
            cells.append(f"{out_m:,.0f}" if out_m is not None else "—")
        r.table_row(cells, ws, al)

    # --- Cost ---
    r.subsection(f"{prefix}Cost USD (mean per task)")
    r.line()
    col_hdrs = [f"Cost {lbl}" for lbl in labels]
    ws = [w_domain] + [16] * len(col_hdrs)
    al = ["l"] + ["r"] * len(col_hdrs)
    r.table_row(["Domain"] + col_hdrs, ws, al)
    r.table_sep(ws, al)
    for domain in all_domains:
        cells = [domain]
        for label in labels:
            tasks = domain_data[label].get(domain, [])
            if not tasks:
                cells.append("—")
                continue
            c = _domain_agg(tasks)["cost_usd"]["mean"]
            cells.append(f"${c:.4f}" if c is not None else "—")
        r.table_row(cells, ws, al)

    # --- Latency ---
    r.subsection(f"{prefix}Latency (mean seconds per task)")
    r.line()
    col_hdrs = [f"Latency {lbl}" for lbl in labels]
    ws = [w_domain] + [18] * len(col_hdrs)
    al = ["l"] + ["r"] * len(col_hdrs)
    r.table_row(["Domain"] + col_hdrs, ws, al)
    r.table_sep(ws, al)
    for domain in all_domains:
        cells = [domain]
        for label in labels:
            tasks = domain_data[label].get(domain, [])
            if not tasks:
                cells.append("—")
                continue
            lat = _domain_agg(tasks)["latency_seconds"]["mean"]
            cells.append(f"{lat:.1f}s" if lat is not None else "—")
        r.table_row(cells, ws, al)

    # --- Researchers ---
    r.subsection(f"{prefix}Researchers Spawned (mean per task)")
    r.line()
    col_hdrs = [f"Researchers {lbl}" for lbl in labels]
    ws = [w_domain] + [22] * len(col_hdrs)
    al = ["l"] + ["r"] * len(col_hdrs)
    r.table_row(["Domain"] + col_hdrs, ws, al)
    r.table_sep(ws, al)
    for domain in all_domains:
        cells = [domain]
        for label in labels:
            tasks = domain_data[label].get(domain, [])
            if not tasks:
                cells.append("—")
                continue
            ri = _domain_agg(tasks)["researcher_instances"]["mean"]
            cells.append(f"{ri:.1f}" if ri is not None else "—")
        r.table_row(cells, ws, al)

    # --- Searches + URLs ---
    r.subsection(f"{prefix}Search Calls & URLs (mean per task)")
    r.line()
    col_hdrs = []
    for lbl in labels:
        col_hdrs += [f"Searches {lbl}", f"URLs {lbl}"]
    ws = [w_domain] + [22] * len(col_hdrs)
    al = ["l"] + ["r"] * len(col_hdrs)
    r.table_row(["Domain"] + col_hdrs, ws, al)
    r.table_sep(ws, al)
    for domain in all_domains:
        cells = [domain]
        for label in labels:
            tasks = domain_data[label].get(domain, [])
            if not tasks:
                cells += ["—", "—"]
                continue
            agg = _domain_agg(tasks)
            sc = agg["search_calls"]["mean"]
            uc = agg["url_count"]["mean"]
            cells.append(f"{sc:.1f}" if sc is not None else "—")
            cells.append(f"{uc:.1f}" if uc is not None else "—")
        r.table_row(cells, ws, al)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def generate_turns_report(
    model_name: str,
    cfg: dict,
    data: dict,
    task_domains: dict,
    output_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"Generating: {output_path.name}")
    print(f"{'='*60}")

    all_projects = data["projects"]
    projects = {k: all_projects[k] for k in cfg["turn_project_labels"] if k in all_projects}
    report_analysis = data.get("report_analysis", {})

    r = Reporter()
    r.line(f"# {model_name} — Turns Analysis (v1 / v2 / v3)")
    r.line()

    print_overall(r, projects)
    print_report_chars_turns(r, report_analysis, cfg["report_key"])

    r.save(output_path)


def generate_sr_report(
    model_name: str,
    cfg: dict,
    data: dict,
    task_domains: dict,
    output_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"Generating: {output_path.name}")
    print(f"{'='*60}")

    all_projects = data["projects"]
    projects = {k: all_projects[k] for k in cfg["sr_project_labels"] if k in all_projects}
    report_analysis = data.get("report_analysis", {})

    r = Reporter()
    r.line(f"# {model_name} — T1 vs Self-Reflect Comparison")
    r.line()

    print_overall(r, projects)
    print_report_chars_sr(r, report_analysis, cfg["report_key"], cfg["report_sr_key"])

    r.save(output_path)


def generate_domain_report(
    model_name: str,
    cfg: dict,
    data: dict,
    task_domains: dict,
    output_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"Generating: {output_path.name}")
    print(f"{'='*60}")

    all_projects = data["projects"]
    turns_projects = {k: all_projects[k] for k in cfg["turn_project_labels"] if k in all_projects}
    sr_projects    = {k: all_projects[k] for k in cfg["sr_project_labels"]   if k in all_projects}

    r = Reporter()
    r.line(f"# {model_name} — Domain-Level Breakdown")
    r.line()

    r.section("Turns (v1 / v2 / v3)")
    print_domain_analysis(r, turns_projects, task_domains, section_prefix="Turns")

    r.section("T1 vs Self-Reflect")
    print_domain_analysis(r, sr_projects, task_domains, section_prefix="Self-Reflect")

    r.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze ablations trace metrics.")
    parser.add_argument(
        "--input", default=None, metavar="FILE",
        help="Trace metrics JSON. Defaults to ablations/trace_metrics/trace_metrics_deepseekv4flash.json "
             "when --deepseekv4flash is set, otherwise ablations/trace_metrics/trace_metrics_merged.json."
    )
    parser.add_argument(
        "--deepseekv4flash", action="store_true",
        help="Analyse DeepSeek v4 Flash results instead of GPT-4.1 / GPT-4.1-mini.",
    )
    args = parser.parse_args()

    if args.deepseekv4flash:
        model_configs = MODEL_CONFIGS_DEEPSEEK
        default_input = "ablations/trace_metrics/trace_metrics_deepseekv4flash.json"
        out_dir       = ANALYSIS_DEEPSEEK_DIR
        n_reports     = len(model_configs) * 3
    else:
        model_configs = MODEL_CONFIGS_GPT
        default_input = "ablations/trace_metrics/trace_metrics_merged.json"
        out_dir       = ANALYSIS_DIR
        n_reports     = len(model_configs) * 3

    input_path = Path(args.input if args.input else default_input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {input_path}\n"
            "Run extract_trace_metrics.py first and merge the outputs."
        )

    data = json.loads(input_path.read_text())
    task_domains = load_task_domains(TASKS_DIR)
    print(f"Loaded {len(task_domains)} task→domain mappings from {TASKS_DIR}")

    for model_name, cfg in model_configs.items():
        generate_turns_report(model_name, cfg, data, task_domains, cfg["turns_output"])
        generate_sr_report(model_name, cfg, data, task_domains, cfg["sr_output"])
        generate_domain_report(model_name, cfg, data, task_domains, cfg["domain_output"])

    print(f"\nAll {n_reports} reports written to {out_dir}/")


if __name__ == "__main__":
    main()
