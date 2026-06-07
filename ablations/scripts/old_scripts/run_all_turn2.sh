#!/usr/bin/env bash
# Run turn-N (2 or 3) for all tasks in ablations/tasks/, sequentially.
# Expects feedback files at ablations/feedback_<slug>[_<save_name>]/<task_id>_v{N-1}_eval_feedback.txt
# Skips tasks where the vN report already exists (OVERWRITE=1 to force).
# Logs each task to ablations/logs_<slug>[_<save_name>]/run_turn<N>_<task_id>.log
#
# Usage:
#   bash ablations/scripts/run_all_turn2.sh                          # turn-2 (default)
#   bash ablations/scripts/run_all_turn2.sh --turn 3                 # turn-3
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_turn2.sh
#   MODEL="bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0" bash ablations/scripts/run_all_turn2.sh
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all_turn2.sh
#   OVERWRITE=1 bash ablations/scripts/run_all_turn2.sh

set -euo pipefail

# --- Turn config ---
TURN="${TURN:-2}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --turn) TURN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
if [[ "$TURN" != "2" && "$TURN" != "3" ]]; then
    echo "Error: --turn must be 2 or 3"
    exit 1
fi
PREV_TURN=$(( TURN - 1 ))

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"

# Compute slug — mirrors run_all_turn1.sh logic so paths always match:
#   bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
#   anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
#   openai:gpt-4.1-mini                                  → gpt41mini
#   deepseek:deepseek-v4-flash                           → deepseekv4flash
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
    MODEL_SLUG="${MODEL_SLUG//-/}"         # remove hyphens → gpt41mini, gemini2.5pro, deepseekv4flash
fi

# --- Save name (optional suffix for all output folders) ---
SAVE_NAME="${SAVE_NAME:-}"
OVERWRITE="${OVERWRITE:-0}"
MAX_REPORT_TOKENS="${MAX_REPORT_TOKENS:-16000}"
SLUG_SUFFIX="${MODEL_SLUG}${SAVE_NAME:+_${SAVE_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
LOG_DIR="$REPO_ROOT/ablations/logs_${SLUG_SUFFIX}"
REPORTS_DIR="$REPO_ROOT/ablations/reports_${SLUG_SUFFIX}"
FEEDBACK_DIR="${FEEDBACK_DIR:-$REPO_ROOT/ablations/feedback_${SLUG_SUFFIX}}"

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
echo "MaxReportTokens: $MAX_REPORT_TOKENS"
echo "Reports:   $REPORTS_DIR"
echo "Feedback:  $FEEDBACK_DIR  (reads v${PREV_TURN}_eval_feedback.txt)"
echo "Logs:      $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    log_file="$LOG_DIR/run_turn${TURN}_${task_id}.log"
    prev_report="$REPORTS_DIR/${task_id}_v${PREV_TURN}.md"
    cur_report="$REPORTS_DIR/${task_id}_v${TURN}.md"
    feedback_file="$FEEDBACK_DIR/${task_id}_v${PREV_TURN}_eval_feedback.txt"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] Starting $task_id ..."

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

    if [ -f "$cur_report" ] && [ "$OVERWRITE" != "1" ]; then
        echo "  SKIPPED — v${TURN} report exists: $cur_report  (set OVERWRITE=1 to force)"
        ((skipped++)) || true
        continue
    fi

    if uv run python ablations/scripts/run_turn.py \
            --task              "ablations/tasks/${task_id}.json" \
            --feedback          "$feedback_file" \
            --model             "$MODEL" \
            --turn              "$TURN" \
            --max_report_tokens "$MAX_REPORT_TOKENS" \
            ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
            > "$log_file" 2>&1; then
        echo "  OK  -> $log_file"
        ((passed++)) || true
    else
        echo "  FAILED -> check $log_file"
        if grep -q "429\|ResourceExhausted\|Resource exhausted\|RESOURCE_EXHAUSTED\|ThrottlingException\|TooManyRequestsException" "$log_file"; then
            echo ""
            echo "  *** Rate limit detected — stopping. Wait and re-run from $task_id. ***"
            break
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
