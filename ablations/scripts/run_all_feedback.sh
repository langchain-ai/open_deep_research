#!/usr/bin/env bash
# Generate feedback from turn-N evaluations for all tasks in ablations/tasks/, sequentially.
# Expects evaluation files at ablations/evaluations/evaluations_<slug>[_<save_name>]/<task_id>_vN_eval.json
# Writes feedback to ablations/feedback/feedback_<slug>[_<save_name>]/<task_id>_vN_eval_feedback.txt
# Logs each task to ablations/logs/logs_<slug>[_<save_name>]/feedback_turn<N>_<task_id>.log
#
# Usage:
#   bash ablations/scripts/run_all_feedback.sh                   # turn-1 (default)
#   bash ablations/scripts/run_all_feedback.sh --turn 2          # turn-2 evals → turn-3 feedback
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_feedback.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash ablations/scripts/run_all_feedback.sh
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all_feedback.sh
#   MODEL=openai:gpt-4.1-mini SAVE_NAME=ablation_1 bash ablations/scripts/run_all_feedback.sh

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
SLUG_SUFFIX="${MODEL_SLUG}${SAVE_NAME:+_${SAVE_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
EVALS_DIR="$REPO_ROOT/ablations/evaluations/evaluations_${SLUG_SUFFIX}"
FEEDBACK_DIR="$REPO_ROOT/ablations/feedback/feedback_${SLUG_SUFFIX}"
LOG_DIR="$REPO_ROOT/ablations/logs/logs_${SLUG_SUFFIX}"

mkdir -p "$LOG_DIR" "$FEEDBACK_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Generating feedback for $total tasks (from turn-$TURN evaluations)"
echo "Model:     $MODEL  (slug: $MODEL_SLUG)"
echo "Save name: ${SAVE_NAME:-<none>}"
echo "Evals:     $EVALS_DIR  (reading v${TURN}_eval.json)"
echo "Output:    $FEEDBACK_DIR  (writing v${TURN}_eval_feedback.txt)"
echo "Logs:      $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    eval_file="$EVALS_DIR/${task_id}_v${TURN}_eval.json"
    log_file="$LOG_DIR/feedback_turn${TURN}_${task_id}.log"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] $task_id ..."

    if [ ! -f "$eval_file" ]; then
        echo "  SKIPPED — eval file not found: $eval_file"
        ((skipped++)) || true
        continue
    fi

    feedback_file="$FEEDBACK_DIR/${task_id}_v${TURN}_eval_feedback.txt"
    if [ -f "$feedback_file" ]; then
        echo "  SKIPPED — feedback already exists: $feedback_file"
        ((skipped++)) || true
        continue
    fi

    if uv run python ablations/scripts/new_generate_feedback.py \
            --evaluation "$eval_file" \
            --task "$task_file" \
            --model "$MODEL" \
            ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
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
