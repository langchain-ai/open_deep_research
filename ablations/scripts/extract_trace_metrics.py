"""Extract token usage, cost, and latency metrics from LangSmith traces.

Iterates over top-level runs in each hard-coded project, filters out errored
runs, applies name-substring filters where needed, and writes a JSON summary.

Two modes:
  Normal:    extract all metrics (tokens, cost, latency, phase breakdown)
  Discovery: sample one run per project and dump all child node names/types
             to node_names_discovery.txt so phase mappings can be verified.

Usage:
    uv run python ablations/scripts/extract_trace_metrics.py
    uv run python ablations/scripts/extract_trace_metrics.py --discover
    uv run python ablations/scripts/extract_trace_metrics.py --output ablations/trace_metrics/trace_metrics.json
    uv run python ablations/scripts/extract_trace_metrics.py --verbose
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from langsmith import Client  # noqa: E402


# ---------------------------------------------------------------------------
# Project config  (hard-coded)
# ---------------------------------------------------------------------------
# Each entry: (label, project_name, name_contains_filter | None, is_turn2)
#   name_contains_filter — if set, only keep root runs whose name contains this
#   is_turn2             — enables critic phase buckets
#
# Turn-1 and Turn-2 data already extracted (see draco_eval/scripts/extract_trace_metrics.py).
# Only Turn-3 and self-reflect are fetched here.

PROJECTS = [
    # --- DeepSeek v4 -------------------------------------------------------
    # ("deepseek_turn1",        "50-test-deepseek-v4-turn1",        None,           False),
    ("deepseek_self_reflect", "50-test-deepseek-v4-self-reflect-new", "self_reflect", False),
    # ("deepseek_turn2",        "50-test-deepseek-v4-turn2",        None,           True),
    # ("deepseek_turn3",        "50-test-deepseek-v4-turn3",        None,           True),
]

# Report directories for each model (relative to repo root = parent of ablations/)
_ABLATIONS_DIR = Path(__file__).parent.parent
REPORT_DIRS = {
    "deepseek":              _ABLATIONS_DIR / "reports" / "reports_deepseekv4flash",
    "deepseek_self_reflect": _ABLATIONS_DIR / "reports" / "reports_deepseekv4flash_self_reflect",
}

# ---------------------------------------------------------------------------
# Phase mapping — applied to the NEAREST NAMED ANCESTOR of each LLM leaf
# ---------------------------------------------------------------------------
PHASE_MAP = {
    "clarify_with_user":       "scoping",
    "write_research_brief":    "scoping",
    "supervisor":              "research",
    "researcher":              "research",
    "compress_research":       "research",
    "final_report_generation": "writing",
}

# LangChain/LangGraph internal wrapper names — skip when walking parent chain
_INTERNAL_NAMES = {
    "_ConfigurableModel", "RunnableSequence", "RunnableLambda",
    "LangGraph", "__start__", "ChatOpenAI",
}

# ---------------------------------------------------------------------------
# Count metrics — name sets
# ---------------------------------------------------------------------------
RESEARCHER_INSTANCE_NAMES = {"compress_research"}
RESEARCHER_REACT_NAMES = {"researcher"}
SEARCH_TOOL_NAMES = {"tavily_search", "web_search", "search", "exa_search", "bing_search"}
INTERNAL_TOOL_NAMES = {"think_tool", "ResearchComplete"}

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r'https?://[^\s\'"<>()\[\]{}]+')

def _extract_urls(obj) -> set[str]:
    """Recursively extract all unique URLs from any nested dict/list/str."""
    urls: set[str] = set()
    _walk(obj, urls)
    return urls

def _walk(obj, urls: set[str]) -> None:
    if isinstance(obj, str):
        for m in _URL_RE.findall(obj):
            urls.add(m.rstrip(".,;:!?\"'"))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, urls)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, urls)

def _task_id_from_run_name(run_name: str) -> str:
    """'task_001_turn1' → 'task_001',  'task_098_turn2' → 'task_098',
       'task_005_self_reflect' → 'task_005'."""
    s = re.sub(r'_turn\d+$', '', run_name)
    s = re.sub(r'_self_reflect$', '', s)
    return s

# ---------------------------------------------------------------------------
# Report characteristics — regex patterns
# ---------------------------------------------------------------------------
_INLINE_LINK_RE = re.compile(r'\[([^\]]*)\]\((https?://[^\s)]+)\)')
_BARE_URL_RE    = re.compile(r'(?<!\()(https?://[^\s\)>\]]+)')
_SECTION_RE     = re.compile(r'^#{1,6}\s', re.MULTILINE)
_MD_STRIP_RE    = re.compile(r'[#*_`>\[\]()!|~^]')

def _report_stats(text: str) -> dict:
    """Word count, unique citation count, section count for one markdown report."""
    section_count = len(_SECTION_RE.findall(text))

    urls: set[str] = set()
    for m in _INLINE_LINK_RE.finditer(text):
        urls.add(m.group(2).rstrip(".,)"))
    for m in _BARE_URL_RE.finditer(text):
        urls.add(m.group(1).rstrip(".,)"))
    citation_count = len(urls)

    clean = _MD_STRIP_RE.sub(' ', text)
    word_count = len(clean.split())

    return {
        "word_count":     word_count,
        "citation_count": citation_count,
        "section_count":  section_count,
    }

# ---------------------------------------------------------------------------
# Turn-pair config for URL overlap analysis
# ---------------------------------------------------------------------------
# Each entry: (label, turn1_project_label, turn2_project_label)
# URL overlap pairs — turn-1/turn-2 already have their data in draco_eval trace_metrics.json.
# Here we only compare turn-3 vs self-reflect (both fetched in this run).
TURN_PAIRS = [
    ("deepseek_t3_sr",  "deepseek_turn3",  "deepseek_self_reflect"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_error(run) -> bool:
    return bool(run.error)


def _tokens_cost(run) -> tuple[int, int, float]:
    return (
        run.prompt_tokens or 0,
        run.completion_tokens or 0,
        float(run.total_cost or 0.0),
    )


def _latency_seconds(run) -> float | None:
    if run.end_time is None or run.start_time is None:
        return None
    start = run.start_time
    end = run.end_time
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - start).total_seconds()


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "total": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "total": round(sum(values), 4),
        "n": len(values),
    }


def _fetch_root_runs(
    client: Client,
    project_name: str,
    name_filter: str | None,
    verbose: bool,
) -> list:
    """Fetch, filter-on-error, and filter-by-name root runs for a project."""
    if verbose:
        print(f"\n[{project_name}] Fetching top-level runs …")

    all_runs = list(tqdm(client.list_runs(project_name=project_name, is_root=True),
                         desc=f"  {project_name} | root runs", unit=" runs", leave=True))

    if verbose:
        print(f"[{project_name}] Total runs fetched: {len(all_runs)}")

    errored = [r for r in all_runs if _has_error(r)]
    runs = [r for r in all_runs if not _has_error(r)]

    if verbose and errored:
        print(f"[{project_name}] Dropped {len(errored)} errored run(s): "
              + ", ".join(r.name or str(r.id) for r in errored))

    if name_filter:
        before = len(runs)
        runs = [r for r in runs if name_filter in (r.name or "")]
        if verbose:
            dropped = before - len(runs)
            if dropped:
                print(f"[{project_name}] Dropped {dropped} run(s) not matching "
                      f"name filter '{name_filter}'.")

    if verbose:
        print(f"[{project_name}] Retained {len(runs)} run(s) after filtering.")

    # Deduplicate: if the same task ran more than once, keep only the latest run
    before_dedup = len(runs)
    task_latest: dict[str, object] = {}
    for r in runs:
        tid = _task_id_from_run_name(r.name or str(r.id))
        existing = task_latest.get(tid)
        if existing is None or (
            r.start_time is not None
            and (existing.start_time is None or r.start_time > existing.start_time)  # type: ignore[union-attr]
        ):
            task_latest[tid] = r
    runs = list(task_latest.values())
    deduped = before_dedup - len(runs)
    if deduped:
        print(f"[{project_name}] Deduplicated {deduped} run(s) — kept latest by start_time.")

    # Sanity check: each project should have exactly 50 tasks
    if len(runs) != 50:
        print(f"WARNING [{project_name}]: expected 50 runs, got {len(runs)} after filtering.")
    elif verbose:
        print(f"[{project_name}] 50-run check passed.")

    return runs


# ---------------------------------------------------------------------------
# Discovery pass
# ---------------------------------------------------------------------------

def run_discovery(client: Client, discovery_output: str = "node_names_discovery.txt") -> None:
    lines: list[str] = []

    for label, project_name, name_filter, is_turn2 in PROJECTS:
        lines.append("=" * 70)
        lines.append(f"PROJECT : {project_name}  (label={label}, turn2={is_turn2})")
        lines.append("=" * 70)

        if project_name is None:
            lines.append("  (project_name not set — skipped)\n")
            continue

        runs = _fetch_root_runs(client, project_name, name_filter, verbose=False)

        if not runs:
            lines.append("  (no retained runs found)\n")
            continue

        sample = runs[0]
        lines.append(f"Sample run : {sample.name or sample.id}\n")

        trace_map, _ = _build_trace_map(client, project_name)
        children = sorted(
            trace_map.get(str(sample.id), []),
            key=lambda r: r.start_time or r.id,
        )

        lines.append(f"  {'run_type':<12} {'name':<50} {'phase_guess'}")
        lines.append(f"  {'-'*12} {'-'*50} {'-'*15}")
        for c in children:
            name = c.name or str(c.id)
            rtype = c.run_type or "?"
            phase = PHASE_MAP.get(name, "other")
            lines.append(f"  {rtype:<12} {name:<50} {phase}")

        lines.append("")

    output_path = Path(discovery_output)
    output_path.write_text("\n".join(lines))
    print(f"[Discovery] Written to {output_path}")
    print("Review the 'phase_guess' column and adjust PHASE_MAP if needed.")


# ---------------------------------------------------------------------------
# Phase breakdown (in-memory, no extra API calls)
# ---------------------------------------------------------------------------

def _build_trace_map(client: Client, project_name: str) -> tuple[dict[str, list], dict[str, list]]:
    trace_map: dict[str, list] = defaultdict(list)
    for r in tqdm(client.list_runs(project_name=project_name),
                  desc=f"  {project_name} | all runs", unit=" runs", leave=True):
        tid = str(r.trace_id) if r.trace_id else str(r.id)
        if tid != str(r.id):
            trace_map[tid].append(r)

    tool_trace_map: dict[str, list] = defaultdict(list)
    for r in tqdm(client.list_runs(project_name=project_name, run_type="tool", include_inputs=True),
                  desc=f"  {project_name} | tool runs", unit=" runs", leave=True):
        tid = str(r.trace_id) if r.trace_id else str(r.id)
        if tid != str(r.id):
            tool_trace_map[tid].append(r)

    return trace_map, tool_trace_map


def _nearest_named_ancestor(run_id: str, run_id_map: dict) -> str:
    r = run_id_map.get(run_id)
    if r is None:
        return "other"
    pid = str(r.parent_run_id) if r.parent_run_id else None
    while pid and pid in run_id_map:
        parent = run_id_map[pid]
        name = parent.name or ""
        if name and name not in _INTERNAL_NAMES:
            return name
        pid = str(parent.parent_run_id) if parent.parent_run_id else None
    return "other"


def _phase_breakdown_from_map(
    trace_map: dict[str, list],
    root_run,
) -> dict[str, dict]:
    child_runs = trace_map.get(str(root_run.id), [])

    run_id_map = {str(r.id): r for r in child_runs}
    run_id_map[str(root_run.id)] = root_run

    phase_acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )

    for r in child_runs:
        if r.run_type != "llm":
            continue
        ancestor_name = _nearest_named_ancestor(str(r.id), run_id_map)
        phase = PHASE_MAP.get(ancestor_name, "other")
        in_tok, out_tok, cost = _tokens_cost(r)
        phase_acc[phase]["input_tokens"]  += in_tok
        phase_acc[phase]["output_tokens"] += out_tok
        phase_acc[phase]["cost_usd"]      += cost

    return {p: dict(v) for p, v in phase_acc.items()}


# ---------------------------------------------------------------------------
# Count metrics
# ---------------------------------------------------------------------------

def _compute_counts(child_runs: list, tool_runs: list) -> tuple[dict[str, int], set[str]]:
    researcher_instances   = 0
    researcher_react_iters = 0
    search_calls           = 0
    external_tool_calls    = 0
    tool_failures          = 0
    urls: set[str] = set()

    for r in child_runs:
        name = r.name or ""
        if name in RESEARCHER_INSTANCE_NAMES:
            researcher_instances += 1
        if name in RESEARCHER_REACT_NAMES:
            researcher_react_iters += 1
        if name in SEARCH_TOOL_NAMES:
            search_calls += 1

    for r in tool_runs:
        name = r.name or ""
        if name in INTERNAL_TOOL_NAMES:
            urls |= _extract_urls(r.outputs)
            continue
        external_tool_calls += 1
        if r.error:
            tool_failures += 1
        urls |= _extract_urls(r.inputs)
        urls |= _extract_urls(r.outputs)

    counts = {
        "researcher_instances":   researcher_instances,
        "researcher_react_iters": researcher_react_iters,
        "search_calls":           search_calls,
        "external_tool_calls":    external_tool_calls,
        "tool_failures":          tool_failures,
        "tool_failure_rate":      round(tool_failures / external_tool_calls * 100, 2)
                                  if external_tool_calls else 0.0,
    }
    return counts, urls


def _agg_int(values: list[int]) -> dict:
    if not values:
        return {"mean": None, "median": None, "max": None, "total": None, "n": 0}
    return {
        "mean":   round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "max":    max(values),
        "total":  sum(values),
        "n":      len(values),
    }


# ---------------------------------------------------------------------------
# Per-project extraction
# ---------------------------------------------------------------------------

def extract_project_metrics(
    client: Client,
    project_name: str,
    name_filter: str | None,
    is_turn2: bool,
    verbose: bool = False,
) -> dict:
    runs = _fetch_root_runs(client, project_name, name_filter, verbose)

    trace_map, tool_trace_map = _build_trace_map(client, project_name)

    per_task: dict[str, dict] = {}
    phase_costs_across_tasks: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        run_name = run.name or str(run.id)
        in_tok, out_tok, cost = _tokens_cost(run)
        latency = _latency_seconds(run)

        child_runs = trace_map.get(str(run.id), [])
        tool_runs  = tool_trace_map.get(str(run.id), [])
        phase_data = _phase_breakdown_from_map(trace_map, run)
        counts, urls = _compute_counts(child_runs, tool_runs)

        task_cost = cost or sum(v["cost_usd"] for v in phase_data.values())
        phase_with_pct: dict[str, dict] = {}
        for phase, vals in phase_data.items():
            pct = round(vals["cost_usd"] / task_cost * 100, 2) if task_cost else None
            phase_with_pct[phase] = {**vals, "cost_usd": round(vals["cost_usd"], 6), "pct_of_task": pct}
            phase_costs_across_tasks[phase].append(vals["cost_usd"])

        per_task[run_name] = {
            "run_id": str(run.id),
            "input_tokens":          in_tok,
            "output_tokens":         out_tok,
            "total_tokens":          in_tok + out_tok,
            "cost_usd":              round(cost, 6),
            "latency_seconds":       round(latency, 2) if latency is not None else None,
            "researcher_instances":  counts["researcher_instances"],
            "researcher_react_iters":counts["researcher_react_iters"],
            "search_calls":          counts["search_calls"],
            "external_tool_calls":   counts["external_tool_calls"],
            "tool_failures":         counts["tool_failures"],
            "tool_failure_rate":     counts["tool_failure_rate"],
            "url_count":             len(urls),
            "urls":                  sorted(urls),
            "phases":                phase_with_pct,
        }

    in_toks   = [v["input_tokens"]        for v in per_task.values()]
    out_toks  = [v["output_tokens"]       for v in per_task.values()]
    tot_toks  = [v["total_tokens"]        for v in per_task.values()]
    costs     = [v["cost_usd"]            for v in per_task.values()]
    latencies = [v["latency_seconds"]     for v in per_task.values()
                 if v["latency_seconds"] is not None]
    researcher_inst  = [v["researcher_instances"]   for v in per_task.values()]
    researcher_react = [v["researcher_react_iters"] for v in per_task.values()]
    search_calls_all = [v["search_calls"]           for v in per_task.values()]
    url_counts       = [v["url_count"]              for v in per_task.values()]
    ext_tool_all     = [v["external_tool_calls"]    for v in per_task.values()]
    tool_fail_all    = [v["tool_failures"]          for v in per_task.values()]
    total_calls      = sum(ext_tool_all)
    total_fails      = sum(tool_fail_all)
    total_cost_all   = sum(costs)

    phase_aggregates: dict[str, dict] = {}
    for phase, phase_cost_list in phase_costs_across_tasks.items():
        mean_cost = statistics.mean(phase_cost_list) if phase_cost_list else 0.0
        total_phase_cost = sum(phase_cost_list)
        pct_of_total = round(total_phase_cost / total_cost_all * 100, 2) if total_cost_all else None
        phase_aggregates[phase] = {
            "mean_cost_usd":     round(mean_cost, 6),
            "total_cost_usd":    round(total_phase_cost, 6),
            "pct_of_total_cost": pct_of_total,
            "n_tasks":           len(phase_cost_list),
        }

    return {
        "project":           project_name,
        "name_filter":       name_filter,
        "n_runs":            len(per_task),
        "per_task":          per_task,
        "aggregates": {
            "input_tokens":        _agg(in_toks),
            "output_tokens":       _agg(out_toks),
            "total_tokens":        _agg(tot_toks),
            "cost_usd":            _agg(costs),
            "latency_seconds":     _agg(latencies),
            "researcher_instances":   _agg_int(researcher_inst),
            "researcher_react_iters": _agg_int(researcher_react),
            "search_calls":           _agg_int(search_calls_all),
            "url_count":              _agg_int(url_counts),
            "external_tool_calls":    _agg_int(ext_tool_all),
            "tool_failures":          _agg_int(tool_fail_all),
        },
        "tool_failure_summary": {
            "total_tool_calls":    total_calls,
            "total_tool_failures": total_fails,
            "failure_rate_pct":    round(total_fails / total_calls * 100, 2) if total_calls else 0.0,
        },
        "phase_aggregates":  phase_aggregates,
    }


# ---------------------------------------------------------------------------
# Report characteristics (word count, citations, sections)
# ---------------------------------------------------------------------------

def analyze_reports() -> dict:
    """Read v1, v2, v3 and self-reflect v2 report files and compute text characteristics.

    Report dirs:
      gpt41              → reports_gpt4.1/   (has v1, v2, v3)
      gpt41_self_reflect → reports_gpt4.1_self_reflect/  (has v2 = self-reflect revision)
      mini               → reports_gpt4.1mini/  (has v1, v2, v3)
      mini_self_reflect  → reports_gpt4.1mini_self_reflect/ (has v2)

    Returns per-model stats with per-version aggregates and deltas.
    """
    results: dict[str, dict] = {}

    for model_key, report_dir in REPORT_DIRS.items():
        if not report_dir.exists():
            results[model_key] = {"error": f"Directory not found: {report_dir}"}
            continue

        is_self_reflect = "self_reflect" in model_key

        # For self-reflect dirs: only v2 files exist (named task_XXX_v2.md)
        # For base dirs: v1, v2, v3 files exist
        v_files: dict[str, dict[str, Path]] = {}  # {version: {task_id: path}}

        if is_self_reflect:
            versions = ["v2"]
        else:
            versions = ["v1", "v2", "v3"]

        for v in versions:
            v_files[v] = {
                f.stem.replace(f"_{v}", ""): f
                for f in report_dir.glob(f"*_{v}.md")
            }

        # Intersect task IDs across all available versions
        all_task_sets = [set(v_files[v].keys()) for v in versions]
        task_ids = sorted(set.intersection(*all_task_sets)) if all_task_sets else []

        per_task: dict[str, dict] = {}
        agg_data: dict[str, list[int]] = {v: {"words": [], "cites": [], "secs": []}
                                           for v in versions}  # type: ignore[assignment]
        agg_data = {v: {"words": [], "cites": [], "secs": []} for v in versions}

        for task_id in task_ids:
            task_entry: dict[str, dict] = {}
            for v in versions:
                stats = _report_stats(v_files[v][task_id].read_text(encoding="utf-8"))
                task_entry[v] = stats
                agg_data[v]["words"].append(stats["word_count"])
                agg_data[v]["cites"].append(stats["citation_count"])
                agg_data[v]["secs"].append(stats["section_count"])

            # Deltas: v1→v2, v2→v3 (if applicable)
            deltas: dict[str, dict] = {}
            if "v1" in task_entry and "v2" in task_entry:
                deltas["v1_to_v2"] = {
                    "word_count":     task_entry["v2"]["word_count"]     - task_entry["v1"]["word_count"],
                    "citation_count": task_entry["v2"]["citation_count"] - task_entry["v1"]["citation_count"],
                    "section_count":  task_entry["v2"]["section_count"]  - task_entry["v1"]["section_count"],
                }
            if "v2" in task_entry and "v3" in task_entry:
                deltas["v2_to_v3"] = {
                    "word_count":     task_entry["v3"]["word_count"]     - task_entry["v2"]["word_count"],
                    "citation_count": task_entry["v3"]["citation_count"] - task_entry["v2"]["citation_count"],
                    "section_count":  task_entry["v3"]["section_count"]  - task_entry["v2"]["section_count"],
                }
            if "v1" in task_entry and "v3" in task_entry:
                deltas["v1_to_v3"] = {
                    "word_count":     task_entry["v3"]["word_count"]     - task_entry["v1"]["word_count"],
                    "citation_count": task_entry["v3"]["citation_count"] - task_entry["v1"]["citation_count"],
                    "section_count":  task_entry["v3"]["section_count"]  - task_entry["v1"]["section_count"],
                }
            per_task[task_id] = {**task_entry, "deltas": deltas}

        def _m(lst: list) -> float | None:
            return round(statistics.mean(lst), 2) if lst else None

        agg_out: dict[str, dict] = {}
        for v in versions:
            agg_out[v] = {
                "word_count_mean":     _m(agg_data[v]["words"]),
                "citation_count_mean": _m(agg_data[v]["cites"]),
                "section_count_mean":  _m(agg_data[v]["secs"]),
            }

        # Overall delta aggregates
        delta_agg: dict[str, dict] = {}
        if "v1" in agg_data and "v2" in agg_data and task_ids:
            delta_agg["v1_to_v2"] = {
                "word_count_delta_mean":     _m([pt["v2"]["word_count"]     - pt["v1"]["word_count"]     for pt in per_task.values() if "v1" in pt and "v2" in pt]),
                "citation_count_delta_mean": _m([pt["v2"]["citation_count"] - pt["v1"]["citation_count"] for pt in per_task.values() if "v1" in pt and "v2" in pt]),
                "section_count_delta_mean":  _m([pt["v2"]["section_count"]  - pt["v1"]["section_count"]  for pt in per_task.values() if "v1" in pt and "v2" in pt]),
            }
        if "v2" in agg_data and "v3" in agg_data and task_ids:
            delta_agg["v2_to_v3"] = {
                "word_count_delta_mean":     _m([pt["v3"]["word_count"]     - pt["v2"]["word_count"]     for pt in per_task.values() if "v2" in pt and "v3" in pt]),
                "citation_count_delta_mean": _m([pt["v3"]["citation_count"] - pt["v2"]["citation_count"] for pt in per_task.values() if "v2" in pt and "v3" in pt]),
                "section_count_delta_mean":  _m([pt["v3"]["section_count"]  - pt["v2"]["section_count"]  for pt in per_task.values() if "v2" in pt and "v3" in pt]),
            }

        results[model_key] = {
            "n_tasks":    len(task_ids),
            "per_task":   per_task,
            "aggregates": agg_out,
            "delta_aggregates": delta_agg,
        }

    return results


# ---------------------------------------------------------------------------
# URL overlap analysis
# ---------------------------------------------------------------------------

def compute_url_overlap(results: dict[str, dict]) -> dict[str, dict]:
    """For each turn-pair, compute per-task URL overlap."""
    overlap_results: dict[str, dict] = {}

    for pair_label, t1_key, t2_key in TURN_PAIRS:
        t1_data = results.get(t1_key, {})
        t2_data = results.get(t2_key, {})

        if not t1_data or not t2_data:
            overlap_results[pair_label] = {"error": "one or both projects missing"}
            continue

        t1_by_task = {_task_id_from_run_name(k): set(v["urls"]) for k, v in t1_data.get("per_task", {}).items()}
        t2_by_task = {_task_id_from_run_name(k): set(v["urls"]) for k, v in t2_data.get("per_task", {}).items()}

        common_tasks = sorted(set(t1_by_task) & set(t2_by_task))

        per_task_overlap: dict[str, dict] = {}
        overlap_ratios, new_counts, dropped_counts = [], [], []

        for task_id in common_tasks:
            t1_urls = t1_by_task[task_id]
            t2_urls = t2_by_task[task_id]
            intersection = t1_urls & t2_urls
            union        = t1_urls | t2_urls
            new_in_t2    = t2_urls - t1_urls
            dropped      = t1_urls - t2_urls
            ratio = round(len(intersection) / len(union), 4) if union else None

            per_task_overlap[task_id] = {
                "t1_url_count":    len(t1_urls),
                "t2_url_count":    len(t2_urls),
                "shared":          len(intersection),
                "new_in_t2":       len(new_in_t2),
                "dropped_in_t2":   len(dropped),
                "overlap_ratio":   ratio,
            }

            if ratio is not None:
                overlap_ratios.append(ratio)
            new_counts.append(len(new_in_t2))
            dropped_counts.append(len(dropped))

        def _mean(lst):
            return round(statistics.mean(lst), 4) if lst else None

        overlap_results[pair_label] = {
            "n_tasks":    len(common_tasks),
            "per_task":   per_task_overlap,
            "aggregates": {
                "overlap_ratio": {
                    "mean":   _mean(overlap_ratios),
                    "median": round(statistics.median(overlap_ratios), 4) if overlap_ratios else None,
                },
                "new_urls_in_t2":     {"mean": _mean(new_counts),     "median": round(statistics.median(new_counts), 2)     if new_counts     else None},
                "dropped_urls_in_t2": {"mean": _mean(dropped_counts), "median": round(statistics.median(dropped_counts), 2) if dropped_counts else None},
            },
        }

    return overlap_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract token/cost/latency metrics from LangSmith traces (ablations)."
    )
    parser.add_argument(
        "--output", default="ablations/trace_metrics/trace_metrics_deepseekv4flash.json", metavar="FILE",
        help="Output JSON file path (default: ablations/trace_metrics/trace_metrics_deepseekv4flash.json)"
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Discovery mode: sample one run per project and dump all child "
             "node names to node_names_discovery.txt, then exit."
    )
    parser.add_argument(
        "--discovery-output", default="node_names_discovery.txt", metavar="FILE",
        help="File to write discovery results to (default: node_names_discovery.txt)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-project filtering details."
    )
    args = parser.parse_args()

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise EnvironmentError("LANGSMITH_API_KEY is not set. Add it to your .env file.")

    client = Client(api_key=api_key)

    if args.discover:
        run_discovery(client, args.discovery_output)
        return

    results: dict[str, dict] = {}
    for label, project_name, name_filter, is_turn2 in PROJECTS:
        if project_name is None:
            print(f"\n[{label}] SKIPPED — project_name not set (fill in PROJECTS config)")
            results[label] = {"n_runs": 0, "skipped": True}
            continue

        print(f"\n[{label}] Processing project: {project_name}", flush=True)
        results[label] = extract_project_metrics(
            client, project_name, name_filter, is_turn2, verbose=args.verbose
        )
        print(f"[{label}] Done — {results[label]['n_runs']} tasks.", flush=True)

    url_overlap     = compute_url_overlap(results)
    report_analysis = analyze_reports()

    output = {
        "projects":        results,
        "url_overlap":     url_overlap,
        "report_analysis": report_analysis,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[Saved] Metrics written to {output_path}")

    active_projects = {k: v for k, v in results.items() if not v.get("skipped")}

    # --- summary table: overall ------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Label':<22} {'n':>4} {'Avg InTok':>10} {'Avg OutTok':>11} "
          f"{'Total $':>9} {'Avg Lat':>9}")
    print("-" * 80)
    for label, data in active_projects.items():
        agg = data["aggregates"]
        avg_in   = agg["input_tokens"]["mean"]    or 0
        avg_out  = agg["output_tokens"]["mean"]   or 0
        tot_cost = agg["cost_usd"]["total"]       or 0
        avg_lat  = agg["latency_seconds"]["mean"] or 0
        print(
            f"{label:<22} {data['n_runs']:>4} "
            f"{avg_in:>10,.0f} {avg_out:>11,.0f} "
            f"${tot_cost:>8.4f} {avg_lat:>8.1f}s"
        )
    print("=" * 80)

    # --- summary table: counts + URLs ------------------------------------
    print("\nCOUNT METRICS (mean / median / max per task)")
    print("=" * 114)
    print(f"{'Label':<22} {'Researchers':>14} {'React iters':>14} {'Searches':>12} {'URLs':>12}")
    print("-" * 114)
    for label, data in active_projects.items():
        agg = data["aggregates"]
        def _fmt(k):
            a = agg[k]
            if a["mean"] is None:
                return "—"
            return f"{a['mean']:.1f}/{a['median']:.1f}/{a['max']}"
        print(f"{label:<22} {_fmt('researcher_instances'):>14} {_fmt('researcher_react_iters'):>14} "
              f"{_fmt('search_calls'):>12} {_fmt('url_count'):>12}")
    print("=" * 114)

    # --- summary table: phase breakdown ----------------------------------
    print("\nPHASE COST BREAKDOWN (mean cost per task  |  % of project total cost)")
    print("=" * 88)
    all_phases = sorted({
        phase
        for data in active_projects.values()
        for phase in data.get("phase_aggregates", {})
    })
    header = f"{'Label':<22}" + "".join(f"  {p:<18}" for p in all_phases)
    print(header)
    print("-" * max(88, len(header)))
    for label, data in active_projects.items():
        pa = data.get("phase_aggregates", {})
        row = f"{label:<22}"
        for phase in all_phases:
            if phase in pa:
                mean_c = pa[phase]["mean_cost_usd"]
                pct    = pa[phase]["pct_of_total_cost"]
                cell   = f"${mean_c:.4f} ({pct}%)"
            else:
                cell = "—"
            row += f"  {cell:<18}"
        print(row)
    print("=" * 88)

    # --- summary table: URL overlap --------------------------------------
    print("\nURL OVERLAP (mean overlap ratio | mean new | mean dropped)")
    print("=" * 72)
    print(f"{'Pair':<20} {'n tasks':>8} {'Overlap ratio':>15} {'New':>10} {'Dropped':>10}")
    print("-" * 72)
    for pair_label, pair_data in url_overlap.items():
        if "error" in pair_data:
            print(f"{pair_label:<20}  (skipped: {pair_data['error']})")
            continue
        agg = pair_data["aggregates"]
        ratio  = agg["overlap_ratio"]["mean"]
        new_u  = agg["new_urls_in_t2"]["mean"]
        drop_u = agg["dropped_urls_in_t2"]["mean"]
        print(
            f"{pair_label:<20} {pair_data['n_tasks']:>8} "
            f"{(ratio or 0):>14.3f} "
            f"{(new_u or 0):>10.1f} "
            f"{(drop_u or 0):>10.1f}"
        )
    print("=" * 72)

    # --- summary table: tool failure rates -------------------------------
    print("\nTOOL FAILURE RATES")
    print("=" * 64)
    print(f"{'Label':<22} {'Total calls':>12} {'Failures':>10} {'Failure %':>10}")
    print("-" * 64)
    for label, data in active_projects.items():
        tfs = data["tool_failure_summary"]
        print(
            f"{label:<22} {tfs['total_tool_calls']:>12} "
            f"{tfs['total_tool_failures']:>10} "
            f"{tfs['failure_rate_pct']:>9.2f}%"
        )
    print("=" * 64)

    # --- summary table: report characteristics ---------------------------
    print("\nREPORT CHARACTERISTICS")
    print("=" * 80)
    for model_key, rdata in report_analysis.items():
        if "error" in rdata:
            print(f"{model_key:<24}  ERROR: {rdata['error']}")
            continue
        print(f"\n  {model_key} ({rdata['n_tasks']} tasks)")
        agg = rdata["aggregates"]
        print(f"  {'Version':<8} {'Words':>10} {'Citations':>10} {'Sections':>10}")
        print("  " + "-" * 42)
        for v, vals in agg.items():
            print(f"  {v:<8} {(vals['word_count_mean'] or 0):>10.1f} "
                  f"{(vals['citation_count_mean'] or 0):>10.1f} "
                  f"{(vals['section_count_mean'] or 0):>10.1f}")
        d_agg = rdata.get("delta_aggregates", {})
        for pair, dvals in d_agg.items():
            print(f"  Δ {pair:<6} {(dvals.get('word_count_delta_mean') or 0):>+10.1f} "
                  f"{(dvals.get('citation_count_delta_mean') or 0):>+10.1f} "
                  f"{(dvals.get('section_count_delta_mean') or 0):>+10.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
