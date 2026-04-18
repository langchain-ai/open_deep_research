#!/usr/bin/env bash
# Compare turn-1 and turn-2 evaluations for all tasks in draco_eval/tasks/.
# Expects: draco_eval/evaluations_<slug>/<task_id>_v1_eval.json
#          draco_eval/evaluations_<slug>/<task_id>_v2_eval.json
# Writes:  draco_eval/analysis_<slug>/<task_id>.json
# Logs:    draco_eval/logs_<slug>/compare_<task_id>.log
#
# Usage:
#   bash draco_eval/scripts/run_all_compare.sh
#   MODEL=openai:gpt-4.1-mini bash draco_eval/scripts/run_all_compare.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash draco_eval/scripts/run_all_compare.sh

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"
MODEL_SLUG="${MODEL##*:}"        # strip provider prefix
MODEL_SLUG="${MODEL_SLUG//-/}"   # remove hyphens

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/draco_eval/tasks"
EVALS_DIR="$REPO_ROOT/draco_eval/evaluations_${MODEL_SLUG}"
LOG_DIR="$REPO_ROOT/draco_eval/logs_${MODEL_SLUG}"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Comparing turn-1 vs turn-2 evaluations for $total tasks"
echo "Model:  $MODEL  (slug: $MODEL_SLUG)"
echo "Evals:  $EVALS_DIR"
echo "Logs:   $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    v1_eval="$EVALS_DIR/${task_id}_v1_eval.json"
    v2_eval="$EVALS_DIR/${task_id}_v2_eval.json"
    log_file="$LOG_DIR/compare_${task_id}.log"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] $task_id ..."

    if [ ! -f "$v1_eval" ]; then
        echo "  SKIPPED — v1 eval not found: $v1_eval"
        ((skipped++)) || true
        continue
    fi

    if [ ! -f "$v2_eval" ]; then
        echo "  SKIPPED — v2 eval not found: $v2_eval"
        ((skipped++)) || true
        continue
    fi

    if uv run python draco_eval/scripts/compare_turns.py \
            --v1    "$v1_eval" \
            --v2    "$v2_eval" \
            --task  "$task_file" \
            --model "$MODEL" \
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
echo "Done. $passed/$total succeeded, $skipped skipped."
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for t in "${failed_tasks[@]}"; do echo "  - $t"; done
fi
echo "============================================================"
