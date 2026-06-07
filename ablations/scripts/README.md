# ablations/scripts — File Reference

Scripts are grouped by the stage of the ablation pipeline they belong to.

---

## 1. Report Generation

**`run_turn.py`** — Run a single task for any turn (1, 2, or 3) via `--turn`. Turn 1 takes the raw task prompt; turns 2/3 take the previous report + evaluator feedback and ask the agent to revise. Output goes to `ablations/reports/reports_<slug>/`.

**`run_all.sh`** — Batch wrapper around `run_turn.py` that loops over all tasks in `ablations/tasks/`. Pass `--turn 1/2/3` to select the turn. Skips tasks where the output report already exists. Logs to `ablations/logs/logs_<slug>/`.

**`run_self_reflect.py`** — Self-reflection baseline for a single task. Reads the v1 report from `ablations/reports/reports_<slug>/` and asks the agent to revise it without any external evaluator feedback — purely through self-critique. Output goes to `ablations/reports/reports_<slug>_self_reflect/`.

**`run_all_self_reflect.sh`** — Batch wrapper around `run_self_reflect.py` for all tasks.

---

## 2. Evaluation

**`evaluate_report.py`** — Evaluate a single report against the task's rubric criteria using an LLM judge (OpenAI or Gemini). Produces a per-criterion `eval.json` with verdicts and a normalized score summary. Output goes to `ablations/evaluations/evaluations_<slug>/`.

**`run_all_eval.sh`** — Batch wrapper around `evaluate_report.py` for all tasks. Pass `--turn 1/2/3` to select which version to evaluate. Reads reports from `ablations/reports/reports_<slug>/`.

---

## 3. Feedback Generation

**`new_generate_feedback.py`** — Generate research-gap-aware feedback for a single task from its eval JSON. Reasons about what the pass/fail pattern reveals about the model's research process, then writes consolidated natural-language feedback targeting FA and BD failures. Uses CQ signals as diagnostics. Saves both the feedback (`feedback.txt`) and the full reasoning (`feedback_full.txt`). Output goes to `ablations/feedback/feedback_<slug>/`.

**`run_all_feedback.sh`** — Batch wrapper around `new_generate_feedback.py` for all tasks. Pass `--turn 1/2` to generate feedback from v1/v2 evals for use in the next turn. Reads from `ablations/evaluations/evaluations_<slug>/`.

---

## 4. Score Analysis

**`compare_turns.py`** — Compare two eval JSONs (e.g. v1 vs v2) for a single task at the per-criterion level. Shows which specific rubric criteria regressed or improved. If you want aggregate statistics across all tasks instead, run `final_evals.py`.

**`run_all_compare.sh`** — Batch wrapper around `compare_turns.py` for all tasks. Produces one JSON per task in `ablations/analysis/analysis_<slug>/`.

**`summarize_evals.py`** — Quick interactive summary for a single model and turn: prints per-task scores and category averages to stdout. Useful during development to check progress. For a full cross-turn, cross-model summary, run `final_evals.py` instead.

**`final_evals.py`** — Definitive summary script. Computes avg normalized scores, pass rates, incorporation and regression rates (v1→v2, v2→v3, v1→v3), and domain breakdowns for all turns and all models. Saves results to `ablations/analysis/analysis_<slug>/`. Run this once all evals are complete.

**`compare_self_reflect.py`** — Compares the self-reflection baseline against the feedback-driven method for a given model: scores, incorporation/regression rates, and domain breakdown side by side. Saves to `ablations/analysis/analysis_<slug>/`.

**`domain_level_analysis.py`** — Detailed domain × turn breakdown covering T1, T2, T3, self-reflect, and feedback. Computes net criteria gain per domain. Saves to `ablations/analysis/analysis_<slug>/domain_level_analysis.json`.

---

## 5. Citation & Text Analysis

**`compare_reports.py`** — Low-level library providing `extract_citations()`, `match_citations()`, and `compute_ngram_overlap()`. Can also be run as a standalone CLI to inspect citation overlap between any two report files. If you want results across all tasks for a model, run `analyse_citations.py` instead.

**`analyse_citations.py`** — Batch citation overlap analysis for all tasks of a given model. Runs three comparisons (v1 vs v2, v1 vs self-reflect, v2 vs v3) and produces a single summary markdown in `ablations/analysis/analyse_citations/`.

---

## 6. Trace Metrics

**`extract_trace_metrics.py`** — Pulls token usage, cost, and latency data from LangSmith traces for all configured projects and writes a raw JSON file (`trace_metrics_<model>.json`) to `ablations/trace_metrics/`. Run this first to collect trace data.

**`analyze_trace_metrics.py`** — Reads the trace metrics JSON from `ablations/trace_metrics/` and generates human-readable markdown reports (turns breakdown, self-reflect comparison, domain breakdown) in `ablations/analysis/analysis_trace_level/` (GPT) or `ablations/analysis/analysis_deepseekv4flash/` (DeepSeek). Run this after `extract_trace_metrics.py`.

---

## `old_scripts/`

Contains superseded scripts kept for reference: `run_turn1.py`, `run_turn2.py`, `run_all_turn1.sh`, `run_all_turn2.sh`. These were replaced by the unified `run_turn.py` and `run_all.sh`.