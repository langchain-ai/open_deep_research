#!/usr/bin/env bash
# Evaluate reports for all tasks in ablations/tasks/, sequentially.
#
# Usage:
#   bash ablations/scripts/run_all_eval.sh --turn 1
#   bash ablations/scripts/run_all_eval.sh --turn 2
#   bash ablations/scripts/run_all_eval.sh --turn 3
#   MODEL=openai:gpt-4.1-mini bash ablations/scripts/run_all_eval.sh --turn 1
#   MODEL=google_vertexai:gemini-2.5-pro bash ablations/scripts/run_all_eval.sh --turn 1
#   SAVE_NAME=ablation_1 bash ablations/scripts/run_all_eval.sh --turn 1
#   MODEL=openai:gpt-4.1-mini SAVE_NAME=ablation_1 bash ablations/scripts/run_all_eval.sh --turn 1
#   JUDGE=gemini bash ablations/scripts/run_all_eval.sh --turn 1   # use Gemini 3 Pro judge (GEM_ prefix)
#   SAVE_NAME=self_reflect bash ablations/scripts/run_all_eval.sh --turn 2   # evaluate self-reflection baseline
#   SKIP_FAILS=1 bash ablations/scripts/run_all_eval.sh --turn 1   # coerce malformed verdicts to UNMET instead of failing

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"

# Compute slug — mirrors run_all_turn1.sh logic so paths always match:
#   bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 → bedrock_claudesonnet45
#   anthropic:claude-haiku-4-5-20251001                  → claudehaiku45
#   openai:gpt-4.1-mini                                  → gpt41mini
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
    MODEL_SLUG="${MODEL_SLUG//-/}"         # remove hyphens → gpt41mini, gemini2.5pro
fi

# --- Judge config ---
JUDGE="${JUDGE:-openai}"         # 'openai' (default) or 'gemini'
SKIP_FAILS="${SKIP_FAILS:-0}"    # set to 1 to coerce malformed verdicts → UNMET
FILE_PREFIX=""
if [[ "$JUDGE" == "gemini" ]]; then
    FILE_PREFIX="GEM_"
fi

# --- Save name (optional suffix for all output folders) ---
SAVE_NAME="${SAVE_NAME:-}"
SLUG_SUFFIX="${MODEL_SLUG}${SAVE_NAME:+_${SAVE_NAME}}"

# --- Parse arguments ---
TURN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --turn) TURN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TURN" != "1" && "$TURN" != "2" && "$TURN" != "3" ]]; then
    echo "Error: --turn must be 1, 2, or 3"
    echo "Usage: bash ablations/scripts/run_all_eval.sh --turn <1|2|3>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/ablations/tasks"
REPORTS_DIR="$REPO_ROOT/ablations/reports/reports_${SLUG_SUFFIX}"
EVALS_DIR="$REPO_ROOT/ablations/evaluations/${FILE_PREFIX}evaluations_${SLUG_SUFFIX}"
LOG_DIR="$REPO_ROOT/ablations/logs/logs_${SLUG_SUFFIX}"

mkdir -p "$LOG_DIR" "$EVALS_DIR"

tasks=("$TASKS_DIR"/*.json)
total=${#tasks[@]}
passed=0
failed=0
skipped=0
failed_tasks=()

echo "============================================================"
echo "Evaluating turn-$TURN reports for $total tasks"
echo "Model:     $MODEL  (slug: $MODEL_SLUG)"
echo "Judge:     $JUDGE${FILE_PREFIX:+ (file prefix: $FILE_PREFIX)}"
echo "Save name: ${SAVE_NAME:-<none>}"
echo "Skip fails: ${SKIP_FAILS} (1 = coerce bad verdicts to UNMET)"
echo "Reports:   $REPORTS_DIR"
echo "Evals:     $EVALS_DIR"
echo "Logs:      $LOG_DIR"
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

    if uv run python ablations/scripts/evaluate_report.py \
            --report "$report_file" \
            --task   "ablations/tasks/${task_id}.json" \
            --output "$eval_file" \
            --model  "$MODEL" \
            --judge  "$JUDGE" \
            ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
            ${SKIP_FAILS:+--skip_fails} \
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

        if uv run python ablations/scripts/evaluate_report.py \
                --report "$report_file" \
                --task   "ablations/tasks/${task_id}.json" \
                --output "$eval_file" \
                --model  "$MODEL" \
                --judge  "$JUDGE" \
                ${SAVE_NAME:+--save_name "$SAVE_NAME"} \
                ${SKIP_FAILS:+--skip_fails} \
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
