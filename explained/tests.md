# tests/ — Evaluation and Debugging Tools

This folder contains tools for measuring output quality and extracting trace data from LangSmith. There are no unit tests here — everything is about evaluating the agent's research reports against benchmarks.

---

## File Map

```
tests/
├── run_evaluate.py              ← main script: runs agent against a benchmark dataset
├── evaluators.py                ← 6 scoring functions that grade a final report
├── prompts.py                   ← prompt strings used by the evaluator LLM
├── pairwise_evaluation.py       ← compares two (or three) agent runs head-to-head
├── supervisor_parallel_evaluation.py  ← checks whether supervisor parallelism is correct
├── extract_langsmith_data.py    ← pulls run results from LangSmith into JSONL files
└── expt_results/                ← saved JSONL outputs from past evaluation runs
    ├── deep_research_bench_gpt-4.1.jsonl
    ├── deep_research_bench_claude4-sonnet.jsonl
    └── deep_research_bench_gpt-5.jsonl
```

---

## `run_evaluate.py` — Main Evaluation Runner

**What it does:** Runs the full agent against every example in a LangSmith dataset ("Deep Research Bench") and scores each output using the six evaluators in `evaluators.py`. Results are logged back to LangSmith as an experiment.

**How to use:**
```bash
python tests/run_evaluate.py
```
You must configure the dataset name, model, and parameters at the top of the file before running.

**Key variables to edit before running:**

| Variable | What to change |
|---|---|
| `dataset_name` | The LangSmith dataset to evaluate against |
| `research_model` | Model for the supervisor and researchers |
| `experiment_prefix` | Label for this run in LangSmith UI |
| `max_concurrency` | How many queries to run in parallel (higher = faster but more API cost) |

**How it works internally:**
1. `target(inputs)` — compiles a fresh graph with `MemorySaver`, invokes it on one dataset example, returns the full final state
2. `client.aevaluate(target, data=..., evaluators=...)` — LangSmith calls `target` on each example, then pipes the output through each evaluator function, logging scores to the experiment

**Imports from:** `evaluators.py` (6 eval functions), `deep_researcher.py` (`deep_researcher_builder`)

---

## `evaluators.py` — Scoring Functions

**What it does:** Six functions that each score a `final_report` on a different dimension. Each function calls `gpt-4.1` with a structured output schema to get a numeric score, then returns it in the format LangSmith expects.

**The six evaluators:**

| Function | Score key(s) | What it measures | Needs reference answer? |
|---|---|---|---|
| `eval_overall_quality` | 6 sub-scores (depth, source quality, rigor, practical value, balance, writing) | Holistic quality across all dimensions | No |
| `eval_relevance` | `relevance_score` | Whether every section of the report is on-topic | No |
| `eval_structure` | `structure_and_cohesiveness_score` | Whether the format matches what was asked (list vs. comparison vs. overview) | No |
| `eval_correctness` | `correctness_score` | Whether the report's claims match a known reference answer | **Yes** — needs `reference_outputs["answer"]` in the dataset |
| `eval_groundedness` | `groundedness_score` | Whether claims in the report are supported by the raw search notes (no hallucination) | No — uses `raw_notes` from agent output |
| `eval_completeness` | `completeness_score` | Whether the report fully answers the research brief and the user question | No |

**How scores are normalised:** All raw scores are integers 1–5. Each evaluator divides by 5 before returning, giving a 0–1 float that LangSmith can plot on a consistent scale.

**Groundedness is special:** Unlike the others, `eval_groundedness` doesn't score the report against a reference — it scores it against the agent's own `raw_notes` (the uncompressed search results stored in state). This is how you detect hallucination: a claim grounded in the raw notes is a claim the agent actually found; an ungrounded claim was invented.

**Imports from:** `prompts.py` (all eval prompt strings), `utils.py` (`get_today_str`)

---

## `prompts.py` — Evaluator Prompts

**What it does:** Stores the prompt strings used by `evaluators.py`. Each prompt is a detailed rubric for one dimension of quality.

**Prompt map:**

| Constant | Used by |
|---|---|
| `OVERALL_QUALITY_PROMPT` | `eval_overall_quality` — 6-dimension rubric with scoring instructions |
| `RELEVANCE_PROMPT` | `eval_relevance` — strict per-section relevance check |
| `STRUCTURE_PROMPT` | `eval_structure` — checks format matches query type (list, comparison, etc.) |
| `CORRECTNESS_PROMPT` | `eval_correctness` — compares report against a reference answer |
| `GROUNDEDNESS_PROMPT` | `eval_groundedness` — extracts claims and checks each against raw context |
| `COMPLETENESS_PROMPT` | `eval_completeness` — checks report covers all points in brief and query |

**Imports from:** Nothing.

---

## `pairwise_evaluation.py` — Head-to-Head Comparison

**What it does:** Compares reports from two or three different agent experiments using Claude Opus 4 as the judge. Useful for deciding whether a change to the agent (e.g. a different model, a prompt tweak) produced better outputs.

**Two modes:**

| Function | Use case |
|---|---|
| `head_to_head_evaluator` | Compare exactly two experiments — returns score `[1, 0]` or `[0, 1]` |
| `free_for_all_evaluator` | Rank three experiments — returns scores `[1, 0.5, 0]` in ranked order |

**How to use:** Edit the experiment name strings at the bottom of the file (`single_agent`, `multi_agent_supervisor`, etc.) to match the experiment names logged in LangSmith, then run:
```bash
python tests/pairwise_evaluation.py
```

The judge model is Claude Opus 4 with extended thinking (`budget_tokens: 16000`) — it is designed to produce careful, reasoned comparisons.

**Imports from:** Nothing in this repo (uses LangSmith SDK directly).

---

## `supervisor_parallel_evaluation.py` — Parallelism Check

**What it does:** A targeted evaluator that checks whether the supervisor correctly parallelised its `ConductResearch` calls. For a given query, the LangSmith dataset contains a `reference_outputs["parallel"]` integer — the expected number of parallel researchers. The evaluator checks whether the supervisor's first tool call batch matches that expected count.

**When to use:** If you change the supervisor prompt or the `max_concurrent_research_units` setting and want to verify the supervisor is actually parallelising correctly, not just running researchers one at a time.

```bash
python tests/supervisor_parallel_evaluation.py
```

**Imports from:** `deep_researcher.py` (`deep_researcher_builder`)

---

## `extract_langsmith_data.py` — Pull Results to JSONL

**What it does:** CLI tool that reads all completed runs from a LangSmith evaluation project and writes them to a `.jsonl` file in `tests/expt_results/`. Useful for offline analysis or submitting to external benchmarks.

**How to use:**
```bash
python tests/extract_langsmith_data.py \
  --project-name "your-langsmith-project-name" \
  --model-name "gpt-4.1" \
  --dataset-name "deep_research_bench"
```
Output: `tests/expt_results/deep_research_bench_gpt-4.1.jsonl`

Each line in the JSONL is: `{"id": "...", "prompt": "...", "article": "<final_report>"}`.

**Imports from:** Nothing in this repo.

---

## `expt_results/` — Saved Evaluation Outputs

Pre-existing JSONL files from past evaluation runs:

| File | Contents |
|---|---|
| `deep_research_bench_gpt-4.1.jsonl` | Reports generated by `gpt-4.1` across the benchmark |
| `deep_research_bench_claude4-sonnet.jsonl` | Reports generated by `claude-sonnet-4` |
| `deep_research_bench_gpt-5.jsonl` | Reports generated by `gpt-5` |

These are reference outputs useful for understanding what quality the benchmark expects, or for running `pairwise_evaluation.py` without re-running the agent.

---

## How the Files Interlink

```
run_evaluate.py
    ├── imports evaluators.py   (6 scoring functions)
    └── imports deep_researcher.py (deep_researcher_builder, to compile fresh graph per run)

evaluators.py
    ├── imports prompts.py      (all eval prompt strings)
    └── imports utils.py        (get_today_str)

supervisor_parallel_evaluation.py
    └── imports deep_researcher.py (deep_researcher_builder)

pairwise_evaluation.py          → no internal imports
extract_langsmith_data.py       → no internal imports
prompts.py                      → no internal imports
```

**The key dependency:** both `run_evaluate.py` and `supervisor_parallel_evaluation.py` import `deep_researcher_builder` (not the compiled `deep_researcher`) so they can compile their own instance with `MemorySaver` as checkpointer, which is required for LangSmith's `aevaluate` to work correctly with thread IDs.
