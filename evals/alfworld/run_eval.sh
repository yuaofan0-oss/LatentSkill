#!/usr/bin/env bash
# run_eval.sh
# Run from the project root:
#   bash evals/alfworld/run_eval.sh <checkpoint-dir> [seen|unseen] [gpu_id]
#
# Example:
#   bash evals/alfworld/run_eval.sh <checkpoint-dir> unseen 0

set -euo pipefail

CHECKPOINT=${1:-${CHECKPOINT:-}}
SPLIT=${2:-"unseen"}
GPU_ID=${3:-"0"}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash evals/alfworld/run_eval.sh <checkpoint-dir> [seen|unseen] [gpu_id]"
    exit 1
fi

case "$SPLIT" in
    seen|unseen) ;;
    *)
        echo "[error] split must be 'seen' or 'unseen', got '$SPLIT'"
        exit 1
        ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/evals/alfworld/results/${CHECKPOINT_NAME}_${SPLIT}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/evals/alfworld/logs"}
ALFWORLD_DATA=${ALFWORLD_DATA:-"${PROJECT_ROOT}/alfworld_data/alfworld"}
ALFWORLD_CONFIG=${ALFWORLD_CONFIG:-"evals/alfworld/config_tw.yaml"}
SKILL_CONTEXT_DIR=${SKILL_CONTEXT_DIR:-"evals/alfworld/skills"}
DEVICE=${DEVICE:-cuda}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/eval_alfworld_${CHECKPOINT_NAME}_${SPLIT}.log"

echo "=========================================="
echo "LatentSkill ALFWorld Evaluation"
echo "Checkpoint : ${CHECKPOINT}"
echo "Split      : ${SPLIT}"
echo "GPU        : ${GPU_ID}"
echo "Output     : ${OUTPUT_DIR}"
echo "=========================================="

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$PROJECT_ROOT" \
nohup python -m evals.alfworld.evaluate \
    --checkpoint "$CHECKPOINT" \
    --config_name models/qwen3_8b \
    --split "$SPLIT" \
    --alfworld_data "$ALFWORLD_DATA" \
    --alfworld_config "$ALFWORLD_CONFIG" \
    --skill_context_dir "$SKILL_CONTEXT_DIR" \
    --max_steps 50 \
    --max_new_tokens 2048 \
    --history_length 5 \
    --context_max_length 4096 \
    --conversation_max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
> "$LOG_FILE" 2>&1 &

echo "Background job started. PID: $!"
echo "Log: tail -f ${LOG_FILE}"
