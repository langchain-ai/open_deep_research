#!/usr/bin/env bash
# Compare turn-1 and turn-2 evaluations for all tasks in ablations/tasks/.
# Expects: ablations/evaluations/evaluations_<slug>[_<save_name>]/<task_id>_v1_eval.json
#          ablations/evaluations/evaluations_<slug>[_<save_name>]/<task_id>_v2_eval.json
# Writes:  ablations/analysis/analysis_<slug>[_<save_name>]/<task_id>.json
# Logs:    ablations/logs/logs_<slug>[_<save_name>]/compare_<task_id>.log
#
# Usage:
#   bash ablations/scripts/run_all_compare.sh
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_compare.sh
#   MODEL=google_vertexai:gemini-2.5-pro bash ablations/scripts/run_all_compare.sh
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all_compare.sh
#   MODEL=openai:gpt-4.1-mini SAVE_NAME=ablation_1 bash ablations/scripts/run_all_compare.sh

set -euo pipefail

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
LOG_DIR="$REPO_ROOT/ablations/logs/logs_${SLUG_SUFFIX}"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Comparing turn-1 vs turn-2 evaluations for $total tasks"
echo "Model:     $MODEL  (slug: $MODEL_SLUG)"
echo "Save name: ${SAVE_NAME:-<none>}"
echo "Evals:     $EVALS_DIR"
echo "Logs:      $LOG_DIR"
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

    if uv run python ablations/scripts/compare_turns.py \
            --v1    "$v1_eval" \
            --v2    "$v2_eval" \
            --task  "$task_file" \
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
