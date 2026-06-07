#!/usr/bin/env bash
# Run turn-1 for all tasks in ablations/tasks/, sequentially.
# Logs each task to ablations/logs_<slug>[_<save_name>]/run_turn1_<task_id>.log
#
# Usage:
#   bash ablations/scripts/run_all_turn1.sh
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_turn1.sh
#   MODEL=openai:gpt-4.1 bash ablations/scripts/run_all_turn1.sh
#   MODEL=anthropic:claude-haiku-4-5-20251001 bash ablations/scripts/run_all_turn1.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash ablations/scripts/run_all_turn1.sh
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all_turn1.sh
#   MODEL=openai:gpt-4.1-mini SAVE_NAME=ablation_1 bash ablations/scripts/run_all_turn1.sh
#
# Rate-limit controls (useful for low-tier API keys):
#   CONCURRENCY=1 ITERATIONS=3 SLEEP=60 MODEL=anthropic:claude-haiku-4-5-20251001 bash ablations/scripts/run_all_turn1.sh

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"

# Compute slug — handles Bedrock multi-colon IDs and Anthropic date-suffixed IDs:
#   bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
#   anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
if [[ "$MODEL" == bedrock:* ]]; then
    _mid="${MODEL#bedrock:}"               # us.anthropic.claude-sonnet-4-5-20250929-v1:0
    _mid="${_mid%%:*}"                     # us.anthropic.claude-sonnet-4-5-20250929-v1
    _name="${_mid##*.}"                    # claude-sonnet-4-5-20250929-v1
    _name="$(echo "$_name" | sed 's/-[0-9]\{8\}.*//')"   # claude-sonnet-4-5
    MODEL_SLUG="bedrock_${_name//-/}"      # bedrock_claudesonnet45
elif [[ "$MODEL" == anthropic:* ]]; then
    _name="${MODEL#anthropic:}"            # claude-haiku-4-5-20251001
    _name="$(echo "$_name" | sed 's/-[0-9]\{8\}.*//')"   # claude-haiku-4-5
    MODEL_SLUG="${_name//-/}"              # claudehaiku45
else
    MODEL_SLUG="${MODEL##*:}"              # strip provider prefix
    MODEL_SLUG="${MODEL_SLUG//-/}"         # remove hyphens → gpt4.1, gemini2.5pro
fi

# --- Rate-limit controls ---
CONCURRENCY="${CONCURRENCY:-5}"         # max_concurrent_research_units
ITERATIONS="${ITERATIONS:-6}"           # max_researcher_iterations
MAX_REPORT_TOKENS="${MAX_REPORT_TOKENS:-16000}"  # final_report_model_max_tokens
SLEEP="${SLEEP:-0}"                     # seconds to wait between tasks (0 = no sleep)

# --- Save name (optional suffix for all output folders) ---
SAVE_NAME="${SAVE_NAME:-}"
SLUG_SUFFIX="${MODEL_SLUG}${SAVE_NAME:+_${SAVE_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
LOG_DIR="$REPO_ROOT/ablations/logs_${SLUG_SUFFIX}"
REPORTS_DIR="$REPO_ROOT/ablations/reports_${SLUG_SUFFIX}"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Running turn-1 for $total tasks"
echo "Model:       $MODEL  (slug: $MODEL_SLUG)"
echo "Save name:   ${SAVE_NAME:-<none>}"
echo "Concurrency: $CONCURRENCY  Iterations: $ITERATIONS  MaxReportTokens: $MAX_REPORT_TOKENS  Sleep: ${SLEEP}s"
echo "Reports:     $REPORTS_DIR"
echo "Logs:        $LOG_DIR"
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

    if uv run python ablations/scripts/run_turn.py \
            --task "ablations/tasks/${task_id}.json" \
            --turn 1 \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY" \
            --iterations  "$ITERATIONS" \
            --max_report_tokens "$MAX_REPORT_TOKENS" \
            ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
            > "$log_file" 2>&1; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"
        # Stop immediately if a 429 quota error is detected.
        if grep -q "429\|ResourceExhausted\|Resource exhausted\|RESOURCE_EXHAUSTED\|rate_limit_error" "$log_file"; then
            echo ""
            echo "  *** 429 Rate limit detected — stopping. ***"
            echo "  *** Try: CONCURRENCY=1 ITERATIONS=3 SLEEP=60 or add a payment method at console.anthropic.com ***"
            echo ""
            break
        fi
        # Retry once if a MALFORMED_FUNCTION_CALL caused the failure.
        if grep -q "MALFORMED_FUNCTION_CALL" "$log_file"; then
            echo "  MALFORMED_FUNCTION_CALL detected — retrying once ..."
            retry_log="$LOG_DIR/run_turn1_${task_id}_retry.log"
            if uv run python ablations/scripts/run_turn.py \
                    --task "ablations/tasks/${task_id}.json" \
                    --turn 1 \
                    --model "$MODEL" \
                    --concurrency "$CONCURRENCY" \
                    --iterations  "$ITERATIONS" \
                    --max_report_tokens "$MAX_REPORT_TOKENS" \
                    ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
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

    if [ "$SLEEP" -gt 0 ]; then
        echo "  sleeping ${SLEEP}s before next task..."
        sleep "$SLEEP"
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
