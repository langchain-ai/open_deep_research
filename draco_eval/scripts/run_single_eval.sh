#!/usr/bin/env bash
# Evaluate v2 reports and compare with v1 for a list of tasks.
# Edit TASKS below to add/remove task IDs.
# Usage: bash draco_eval/scripts/run_single_eval.sh

set -euo pipefail

TASKS=("task_045")  # <-- add task IDs here

for task_id in "${TASKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing $task_id ..."
    echo "============================================================"

    echo "  [1/2] Evaluating v2 report..."
    uv run python draco_eval/scripts/evaluate_report.py \
        --report "draco_eval/reports/${task_id}_v2.md" \
        --task   "draco_eval/tasks/${task_id}.json" \
        --output "draco_eval/evaluations/${task_id}_v2_eval.json"

    echo "  [2/2] Comparing turns..."
    uv run python draco_eval/scripts/compare_turns.py \
        --v1   "draco_eval/evaluations/${task_id}_v1_eval.json" \
        --v2   "draco_eval/evaluations/${task_id}_v2_eval.json" \
        --task "draco_eval/tasks/${task_id}.json"

    echo "  Done: $task_id"
done

echo ""
echo "All tasks complete."
