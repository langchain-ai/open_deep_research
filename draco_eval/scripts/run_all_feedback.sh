#!/usr/bin/env bash
# Generate feedback from turn-1 evaluations for all tasks in draco_eval/tasks/, sequentially.
# Expects evaluation files at draco_eval/evaluations_<slug>/<task_id>_v1_eval.json
# Writes feedback to draco_eval/feedback_<slug>/<task_id>_v1_eval_feedback.txt
# Logs each task to draco_eval/logs_<slug>/feedback_<task_id>.log
#
# Usage:
#   bash draco_eval/scripts/run_all_feedback.sh
#   MODEL=openai:gpt-4.1-mini bash draco_eval/scripts/run_all_feedback.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash draco_eval/scripts/run_all_feedback.sh

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"
MODEL_SLUG="${MODEL##*:}"        # strip provider prefix
MODEL_SLUG="${MODEL_SLUG//-/}"   # remove hyphens

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/draco_eval/tasks"
EVALS_DIR="$REPO_ROOT/draco_eval/evaluations_${MODEL_SLUG}"
FEEDBACK_DIR="$REPO_ROOT/draco_eval/feedback_${MODEL_SLUG}"
LOG_DIR="$REPO_ROOT/draco_eval/logs_${MODEL_SLUG}"

mkdir -p "$LOG_DIR" "$FEEDBACK_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Generating feedback for $total tasks (from turn-1 evaluations)"
echo "Model:  $MODEL  (slug: $MODEL_SLUG)"
echo "Evals:  $EVALS_DIR"
echo "Output: $FEEDBACK_DIR"
echo "Logs:   $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    eval_file="$EVALS_DIR/${task_id}_v1_eval.json"
    log_file="$LOG_DIR/feedback_${task_id}.log"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] $task_id ..."

    if [ ! -f "$eval_file" ]; then
        echo "  SKIPPED — eval file not found: $eval_file"
        ((skipped++)) || true
        continue
    fi

    if uv run python draco_eval/scripts/new_generate_feedback.py \
            --evaluation "$eval_file" \
            --task "$task_file" \
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
