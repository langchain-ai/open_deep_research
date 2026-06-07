#!/usr/bin/env bash
# Self-reflection baseline: revise every v1 report without external feedback.
# Reads from  ablations/reports/reports_<slug>/
# Writes to   ablations/reports/reports_<slug>_self_reflect/
# Logs to     ablations/logs/logs_<slug>_self_reflect/
#
# Usage:
#   bash ablations/scripts/run_all_self_reflect.sh
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_self_reflect.sh
#   OVERWRITE=1 bash ablations/scripts/run_all_self_reflect.sh

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

OVERWRITE="${OVERWRITE:-0}"
MAX_REPORT_TOKENS="${MAX_REPORT_TOKENS:-16000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
REPORTS_DIR="$REPO_ROOT/ablations/reports/reports_${MODEL_SLUG}"
OUTPUT_DIR="$REPO_ROOT/ablations/reports/reports_${MODEL_SLUG}_self_reflect"
LOG_DIR="$REPO_ROOT/ablations/logs/logs_${MODEL_SLUG}_self_reflect"

mkdir -p "$LOG_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Self-reflection baseline for $total tasks"
echo "Model:           $MODEL  (slug: $MODEL_SLUG)"
echo "MaxReportTokens: $MAX_REPORT_TOKENS"
echo "Reads:           $REPORTS_DIR  (v1 reports)"
echo "Writes:          $OUTPUT_DIR  (v2 reports)"
echo "Logs:            $LOG_DIR"
echo "============================================================"

for task_file in "${tasks[@]}"; do
    task_id="$(basename "$task_file" .json)"
    log_file="$LOG_DIR/self_reflect_${task_id}.log"
    v1_report="$REPORTS_DIR/${task_id}_v1.md"
    out_report="$OUTPUT_DIR/${task_id}_v2.md"

    echo ""
    echo "[$((passed + failed + skipped + 1))/$total] Starting $task_id ..."

    if [ ! -f "$v1_report" ]; then
        echo "  SKIPPED — v1 report not found: $v1_report  (run turn-1 first)"
        ((skipped++)) || true
        continue
    fi

    if [ -f "$out_report" ] && [ "$OVERWRITE" != "1" ]; then
        echo "  SKIPPED — self-reflect v2 report exists: $out_report  (set OVERWRITE=1 to force)"
        ((skipped++)) || true
        continue
    fi

    if uv run python ablations/scripts/run_self_reflect.py \
            --task              "ablations/tasks/${task_id}.json" \
            --model             "$MODEL" \
            --max_report_tokens "$MAX_REPORT_TOKENS" \
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
