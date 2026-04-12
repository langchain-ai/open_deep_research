#!/usr/bin/env bash
# Run turn-1 for all tasks in draco_eval/tasks/, sequentially.
# Logs each task to draco_eval/logs/run_turn1_<task_id>.log
# Usage: bash draco_eval/scripts/run_all_turn1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/draco_eval/tasks"
LOG_DIR="$REPO_ROOT/draco_eval/logs"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
failed_tasks=()

echo "============================================================"
echo "Running turn-1 for $total tasks"
echo "Logs: $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    log_file="$LOG_DIR/run_turn1_${task_id}.log"

    echo ""
    echo "[$((passed + failed + 1))/$total] Starting $task_id ..."

    if uv run python draco_eval/scripts/run_turn1.py \
            --task "draco_eval/tasks/${task_id}.json" \
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
echo "Done. $passed/$total succeeded."
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for t in "${failed_tasks[@]}"; do echo "  - $t"; done
fi
echo "============================================================"
