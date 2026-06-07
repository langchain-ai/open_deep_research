# Multi-turn Evaluation of Deep Research Agents

Forked from https://github.com/langchain-ai/open_deep_research

---

## ablations/

This is the main folder for our paper. It contains all scripts, tasks, reports, evaluations, and analysis produced during the study.

### Folder structure

| Folder | Contents |
|---|---|
| `scripts/` | All pipeline scripts — running turns, evaluating reports, generating feedback, comparing turns, citation analysis, trace metric extraction, and shell wrappers |
| `tasks/` | Task definitions (JSON files), one per research topic, used as input to the agent |
| `reports/` | Agent-generated research reports, organised by model and turn (e.g. `reports_gpt4.1/task_001_v1.md`) |
| `evaluations/` | Per-report evaluation scores produced by the judge LLM, organised by model (e.g. `evaluations_gpt4.1/`) |
| `analysis/` | Aggregated analysis outputs — per-model summaries, domain-level breakdowns, self-reflect comparisons, citation analysis, and trace-level metrics |
| `prompts/` | Prompt templates used by the evaluator and feedback generator |
| `trace_metrics/` | Raw LangSmith trace-level metrics extracted from agent runs (excluded from git; stored locally only) |
| `logs/` | Shell script run logs (excluded from git; stored locally only) |
| `feedback/` | Evaluator-generated feedback files passed to the agent on subsequent turns (excluded from git; stored locally only) |
| `fig2_code.py` | Code to reproduce Figure 2 from the paper |
