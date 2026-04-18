#!/usr/bin/env bash
# Evaluate reports for all tasks in draco_eval/tasks/, sequentially.
# Usage:
#   bash draco_eval/scripts/run_all_eval.sh --turn 1
#   bash draco_eval/scripts/run_all_eval.sh --turn 2
#   MODEL=openai:gpt-4.1-mini bash draco_eval/scripts/run_all_eval.sh --turn 1
#   MODEL=google_vertexai:gemini-2.5-pro bash draco_eval/scripts/run_all_eval.sh --turn 1

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"
MODEL_SLUG="${MODEL##*:}"        # strip provider prefix
MODEL_SLUG="${MODEL_SLUG//-/}"   # remove hyphens

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
REPORTS_DIR="$REPO_ROOT/draco_eval/reports_${MODEL_SLUG}"
EVALS_DIR="$REPO_ROOT/draco_eval/evaluations_${MODEL_SLUG}"
LOG_DIR="$REPO_ROOT/draco_eval/logs_${MODEL_SLUG}"

mkdir -p "$LOG_DIR" "$EVALS_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Evaluating turn-$TURN reports for $total tasks"
echo "Model:   $MODEL  (slug: $MODEL_SLUG)"
echo "Reports: $REPORTS_DIR"
echo "Evals:   $EVALS_DIR"
echo "Logs:    $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    report_file="$REPORTS_DIR/${task_id}_v${TURN}.md"
    eval_file="$EVALS_DIR/${task_id}_v${TURN}_eval.json"
    log_file="$LOG_DIR/eval_turn${TURN}_${task_id}.log"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] $task_id ..."

    if [ ! -f "$report_file" ]; then
        echo "  SKIPPED — report not found: $report_file"
        ((skipped++)) || true
        continue
    fi

    if [ -f "$eval_file" ]; then
        echo "  SKIPPED — eval already exists: $eval_file"
        ((skipped++)) || true
        continue
    fi

    if uv run python draco_eval/scripts/evaluate_report.py \
            --report "$report_file" \
            --task   "draco_eval/tasks/${task_id}.json" \
            --output "$eval_file" \
            --model  "$MODEL" \
            > "$log_file" 2>&1; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"
        ((failed++)) || true
        failed_tasks+=("$task_id")
    fi
done

# --- Retry failed tasks once ---
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "Retrying ${#failed_tasks[@]} failed task(s)..."
    echo "============================================================"

    still_failed=()
    for task_id in "${failed_tasks[@]}"; do
        report_file="$REPORTS_DIR/${task_id}_v${TURN}.md"
        eval_file="$EVALS_DIR/${task_id}_v${TURN}_eval.json"
        log_file="$LOG_DIR/eval_turn${TURN}_${task_id}_retry.log"

        echo ""
        echo "[RETRY] $task_id ..."

        if uv run python draco_eval/scripts/evaluate_report.py \
                --report "$report_file" \
                --task   "draco_eval/tasks/${task_id}.json" \
                --output "$eval_file" \
                --model  "$MODEL" \
                > "$log_file" 2>&1; then
            echo "  OK  -> $log_file"
            ((passed++)) || true
            ((failed--)) || true
        else
            echo "  FAILED again -> $log_file"
            still_failed+=("$task_id")
        fi
    done

    if [ ${#still_failed[@]} -gt 0 ]; then
        echo ""
        echo "WARNING: the following tasks failed twice, please check logs:"
        for t in "${still_failed[@]}"; do echo "  - $t"; done
    fi
fi

echo ""
echo "============================================================"
echo "Done. $passed/$total succeeded, $skipped skipped (no report)."
echo "============================================================"
