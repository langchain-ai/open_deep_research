#!/usr/bin/env bash
# Run Turn-3 multimodal probe on one task per domain (10 tasks total).
# Tasks are drawn from draco_eval/analysis/turn3_domain_sample.json.
#
# Usage:
#   bash draco_eval/scripts/run_turn3_sample.sh
#   bash draco_eval/scripts/run_turn3_sample.sh --overwrite
#   MODEL=google_genai:gemini-2.5-pro bash draco_eval/scripts/run_turn3_sample.sh

set -euo pipefail

# --- Model config ---
MODEL="${MODEL:-openai:gpt-4.1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/draco_eval/scripts/run_turn3.py"
EXTRA_ARGS="${1:-}"   # pass --overwrite to re-run existing outputs

TASKS=(
    "task_001"   # Academic
    "task_002"   # Finance
    "task_012"   # General Knowledge
    "task_039"   # Law
    "task_031"   # Medicine
    "task_011"   # Needle in a Haystack
    "task_006"   # Personalized Assistant
    "task_003"   # Shopping/Product Comparison
    "task_021"   # Technology
    "task_018"   # UX Design
)

echo "Turn-3 domain sample run (${#TASKS[@]} tasks)"
echo "Model:     $MODEL"
echo "Repo root: $REPO_ROOT"
echo "========================================"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo ">>> $TASK"
    uv run python "$SCRIPT" --task "$TASK" --reports-model "$MODEL" $EXTRA_ARGS
done

echo ""
echo "========================================"
echo "Done."
