#!/usr/bin/env bash
# Run turn-1 for all tasks in draco_eval/tasks/, sequentially.
# Logs each task to draco_eval/logs_<slug>/run_turn1_<task_id>.log
#
# Usage:
#   bash draco_eval/scripts/run_all_turn1.sh
#   MODEL=openai:gpt-4.1-mini bash draco_eval/scripts/run_all_turn1.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash draco_eval/scripts/run_all_turn1.sh

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"
MODEL_SLUG="${MODEL##*:}"        # strip provider prefix (e.g. 'openai:')
MODEL_SLUG="${MODEL_SLUG//-/}"   # remove hyphens → gpt4.1, gemini2.5pro

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/draco_eval/tasks"
LOG_DIR="$REPO_ROOT/draco_eval/logs_${MODEL_SLUG}"

mkdir -p "$LOG_DIR"

REPORTS_DIR="$REPO_ROOT/draco_eval/reports_${MODEL_SLUG}"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Running turn-1 for $total tasks"
echo "Model:   $MODEL  (slug: $MODEL_SLUG)"
echo "Reports: $REPORTS_DIR"
echo "Logs:    $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    log_file="$LOG_DIR/run_turn1_${task_id}.log"
    report_file="$REPORTS_DIR/${task_id}_v1.md"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] Starting $task_id ..."

    if [ -f "$report_file" ]; then
        echo "  SKIPPED — report already exists: $report_file"
        ((skipped++)) || true
        continue
    fi

    if uv run python draco_eval/scripts/run_turn1.py \
            --task "draco_eval/tasks/${task_id}.json" \
            --model "$MODEL" \
            > "$log_file" 2>&1; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"
        # Stop immediately if a 429 quota error is detected.
        if grep -q "429\|ResourceExhausted\|Resource exhausted\|RESOURCE_EXHAUSTED" "$log_file"; then
            echo ""
            echo "  *** 429 Resource Exhausted detected — stopping. ***"
            echo "  *** Top up your quota then re-run from $task_id.  ***"
            echo ""
            break
        fi
        # Retry once if a MALFORMED_FUNCTION_CALL caused the failure.
        if grep -q "MALFORMED_FUNCTION_CALL" "$log_file"; then
            echo "  MALFORMED_FUNCTION_CALL detected — retrying once ..."
            retry_log="$LOG_DIR/run_turn1_${task_id}_retry.log"
            if uv run python draco_eval/scripts/run_turn1.py \
                    --task "draco_eval/tasks/${task_id}.json" \
                    --model "$MODEL" \
                    > "$retry_log" 2>&1; then
                echo "  OK (retry) -> $retry_log"
                ((passed++)) || true
                continue
            else
                echo "  FAILED again -> $retry_log"
            fi
        fi
        ((failed++)) || true
        failed_tasks+=("$task_id")
    fi
done

echo ""
echo "============================================================"
echo "Done. $passed/$total succeeded, $skipped skipped (report exists)."
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for t in "${failed_tasks[@]}"; do echo "  - $t"; done
fi
echo "============================================================"
