#!/usr/bin/env bash
# Evaluate reports for all tasks in draco_eval/tasks/, sequentially.
# Usage:
#   bash draco_eval/scripts/run_all_eval.sh --turn 1
#   bash draco_eval/scripts/run_all_eval.sh --turn 2

set -euo pipefail

# --- Parse arguments ---
TURN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --turn) TURN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TURN" != "1" && "$TURN" != "2" ]]; then
    echo "Error: --turn must be 1 or 2"
    echo "Usage: bash draco_eval/scripts/run_all_eval.sh --turn <1|2>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/draco_eval/tasks"
REPORTS_DIR="$REPO_ROOT/draco_eval/reports"
EVALS_DIR="$REPO_ROOT/draco_eval/evaluations"
LOG_DIR="$REPO_ROOT/draco_eval/logs"

mkdir -p "$LOG_DIR" "$EVALS_DIR"

SKIP_TASKS=("task_014" "task_039" "task_078")

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Evaluating turn-$TURN reports for $total tasks"
echo "Logs: $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    report_file="$REPORTS_DIR/${task_id}_v${TURN}.md"
    eval_file="draco_eval/evaluations/${task_id}_v${TURN}_eval.json"
    log_file="$LOG_DIR/eval_turn${TURN}_${task_id}.log"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] $task_id ..."

    if [[ " ${SKIP_TASKS[*]} " == *" $task_id "* ]]; then
        echo "  SKIPPED — excluded task"
        ((skipped++)) || true
        continue
    fi

    if [ ! -f "$report_file" ]; then
        echo "  SKIPPED — report not found: $report_file"
        ((skipped++)) || true
        continue
    fi

    if uv run python draco_eval/scripts/evaluate_report.py \
            --report "draco_eval/reports/${task_id}_v${TURN}.md" \
            --task   "draco_eval/tasks/${task_id}.json" \
            --output "$eval_file" \
            > "$log_file" 2>&1; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"
        ((failed++)) || true
        failed_tasks+=("$task_id")
    fi
done

echo ""
echo "============================================================"
echo "Done. $passed/$total succeeded, $skipped skipped (no report)."
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for t in "${failed_tasks[@]}"; do echo "  - $t"; done
fi
echo "============================================================"
