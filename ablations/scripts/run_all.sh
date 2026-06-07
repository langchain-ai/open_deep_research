#!/usr/bin/env bash
# Run any turn (1, 2, or 3) for all tasks in ablations/tasks/, sequentially.
#
# Turn 1: runs the agent on each raw task prompt.
# Turn 2/3: reads the previous report + evaluator feedback and asks the agent to revise.
#
# Skips tasks where the target vN report already exists (OVERWRITE=1 to force).
# Logs each task to ablations/logs/logs_<slug>[_<save_name>]/run_turn<N>_<task_id>.log
#
# Usage:
#   bash ablations/scripts/run_all.sh                              # turn-1, GPT-4.1
#   bash ablations/scripts/run_all.sh --turn 2                     # turn-2
#   bash ablations/scripts/run_all.sh --turn 3                     # turn-3
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all.sh
#   MODEL="bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0" bash ablations/scripts/run_all.sh
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all.sh --turn 2
#   OVERWRITE=1 bash ablations/scripts/run_all.sh --turn 2

set -euo pipefail

# --- Turn config ---
TURN="${TURN:-1}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --turn) TURN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
if [[ "$TURN" != "1" && "$TURN" != "2" && "$TURN" != "3" ]]; then
    echo "Error: --turn must be 1, 2, or 3"
    exit 1
fi
PREV_TURN=$(( TURN - 1 ))

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"

# Compute slug:
#   bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
#   anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
#   openai:gpt-4.1-mini                                  → gpt41mini
#   deepseek:deepseek-v4-flash                           → deepseekv4flash
if [[ "$MODEL" == bedrock:* ]]; then
    _mid="${MODEL#bedrock:}"
    _mid="${_mid%%:*}"
    _name="${_mid##*.}"
    _name="$(echo "$_name" | sed 's/-[0-9]\{8\}.*//')"
    MODEL_SLUG="bedrock_${_name//-/}"
elif [[ "$MODEL" == anthropic:* ]]; then
    _name="${MODEL#anthropic:}"
    _name="$(echo "$_name" | sed 's/-[0-9]\{8\}.*//')"
    MODEL_SLUG="${_name//-/}"
else
    MODEL_SLUG="${MODEL##*:}"
    MODEL_SLUG="${MODEL_SLUG//-/}"
fi

# --- Options ---
SAVE_NAME="${SAVE_NAME:-}"
OVERWRITE="${OVERWRITE:-0}"
MAX_REPORT_TOKENS="${MAX_REPORT_TOKENS:-16000}"
SLUG_SUFFIX="${MODEL_SLUG}${SAVE_NAME:+_${SAVE_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
LOG_DIR="$REPO_ROOT/ablations/logs/logs_${SLUG_SUFFIX}"
REPORTS_DIR="$REPO_ROOT/ablations/reports/reports_${SLUG_SUFFIX}"
FEEDBACK_DIR="$REPO_ROOT/ablations/feedback/feedback_${SLUG_SUFFIX}"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Running turn-$TURN for $total tasks"
echo "Model:     $MODEL  (slug: $MODEL_SLUG)"
echo "Save name: ${SAVE_NAME:-<none>}"
echo "Reports:   $REPORTS_DIR"
if [[ "$TURN" != "1" ]]; then
    echo "Feedback:  $FEEDBACK_DIR  (reads v${PREV_TURN}_eval_feedback.txt)"
fi
echo "Logs:      $LOG_DIR"
echo "MaxReportTokens: $MAX_REPORT_TOKENS"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    log_file="$LOG_DIR/run_turn${TURN}_${task_id}.log"
    cur_report="$REPORTS_DIR/${task_id}_v${TURN}.md"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] Starting $task_id ..."

    # --- Turn-2/3: check prerequisites ---
    if [[ "$TURN" != "1" ]]; then
        prev_report="$REPORTS_DIR/${task_id}_v${PREV_TURN}.md"
        feedback_file="$FEEDBACK_DIR/${task_id}_v${PREV_TURN}_eval_feedback.txt"

        if [ ! -f "$prev_report" ]; then
            echo "  SKIPPED — v${PREV_TURN} report not found: $prev_report  (run turn-${PREV_TURN} first)"
            ((skipped++)) || true
            continue
        fi

        if [ ! -f "$feedback_file" ]; then
            echo "  SKIPPED — feedback not found: $feedback_file"
            ((skipped++)) || true
            continue
        fi
    fi

    # --- Skip if output already exists ---
    if [ -f "$cur_report" ] && [ "$OVERWRITE" != "1" ]; then
        echo "  SKIPPED — v${TURN} report exists: $cur_report  (set OVERWRITE=1 to force)"
        ((skipped++)) || true
        continue
    fi

    # --- Run ---
    if [[ "$TURN" == "1" ]]; then
        uv run python ablations/scripts/run_turn.py \
                --task              "ablations/tasks/${task_id}.json" \
                --turn              1 \
                --model             "$MODEL" \
                --max_report_tokens "$MAX_REPORT_TOKENS" \
                ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
                > "$log_file" 2>&1
        exit_code=$?
    else
        uv run python ablations/scripts/run_turn.py \
                --task              "ablations/tasks/${task_id}.json" \
                --turn              "$TURN" \
                --feedback          "$feedback_file" \
                --model             "$MODEL" \
                --max_report_tokens "$MAX_REPORT_TOKENS" \
                ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
                > "$log_file" 2>&1
        exit_code=$?
    fi

    if [ "$exit_code" -eq 0 ]; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"

        # Stop on rate limit
        if grep -q "429\|ResourceExhausted\|Resource exhausted\|RESOURCE_EXHAUSTED\|ThrottlingException\|TooManyRequestsException\|rate_limit_error" "$log_file"; then
            echo ""
            echo "  *** Rate limit detected — stopping. Wait and re-run from $task_id. ***"
            echo ""
            break
        fi

        # Turn-1 only: retry once on MALFORMED_FUNCTION_CALL
        if [[ "$TURN" == "1" ]] && grep -q "MALFORMED_FUNCTION_CALL" "$log_file"; then
            echo "  MALFORMED_FUNCTION_CALL detected — retrying once ..."
            retry_log="$LOG_DIR/run_turn1_${task_id}_retry.log"
            uv run python ablations/scripts/run_turn.py \
                    --task              "ablations/tasks/${task_id}.json" \
                    --turn              1 \
                    --model             "$MODEL" \
                    --max_report_tokens "$MAX_REPORT_TOKENS" \
                    ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
                    > "$retry_log" 2>&1 && {
                echo "  OK (retry) -> $retry_log"
                ((passed++)) || true
                continue
            } || echo "  FAILED again -> $retry_log"
        fi

        ((failed++)) || true
        failed_tasks+=("$task_id")
    fi
done

echo ""
echo "============================================================"
echo "Done. $passed/$total succeeded, $skipped skipped, $failed failed."
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for t in "${failed_tasks[@]}"; do echo "  - $t"; done
fi
echo "============================================================"