#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

NAME=${NAME:-latentskill_sft_qwen3_8b}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29501}
CONFIG_NAME=${CONFIG_NAME:-models/qwen3_8b}
# ===== Training =====
MODE=${MODE:-train}
SOURCE=${SOURCE:-skill-ift}
NUM_EPOCHS=${NUM_EPOCHS:-10}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
USE_GRADIENT_CHECKPOINT=${USE_GRADIENT_CHECKPOINT:-true}

RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:--1}

WARMUP_STEPS=${WARMUP_STEPS:-400}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

CONTEXT_MAX_LEN=${CONTEXT_MAX_LEN:-4096}
CONVERSATION_MAX_LEN=${CONVERSATION_MAX_LEN:-4096}

EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-50}
# ===== MetaLoRA / Hypernetwork =====
TYPE=${TYPE:-transformer}
NUM_LAYERS=${NUM_LAYERS:-4}
METHOD=${METHOD:-rl}
LORA_R=${LORA_R:-8}
METALORA_R=${METALORA_R:-128}

if [ "$MODE" != "train" ]; then
    echo "Unsupported MODE='$MODE'. The SFT script expects MODE=train." >&2
    exit 2
fi

if [ "$SOURCE" != "skill-ift" ]; then
    echo "Unsupported SOURCE='$SOURCE'. The SFT script expects SOURCE=skill-ift." >&2
    exit 2
fi

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES

# ===== Logging and checkpoints =====
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE="$LOG_DIR/sft_$NAME.log"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$NAME/$MODE"
PRETRAIN_CHECKPOINT_NAME=${PRETRAIN_CHECKPOINT_NAME:-latentskill_pretrain_qwen3_8b}

EXTRA_OVERRIDES=()
[ -n "${MODEL_PATH:-}" ] && EXTRA_OVERRIDES+=(paths.model_path="$MODEL_PATH")
[ -n "${DATA_ROOT:-}" ] && EXTRA_OVERRIDES+=(data.root_dir="$DATA_ROOT")
[ -n "${CHECKPOINT_ROOT:-}" ] && EXTRA_OVERRIDES+=(paths.checkpoint_root="$CHECKPOINT_ROOT")
[ -n "${PRETRAIN_CHECKPOINT_DIR:-}" ] && EXTRA_OVERRIDES+=(paths.pretrain_checkpoint_dir="$PRETRAIN_CHECKPOINT_DIR")
[ -n "${PRETRAIN_CHECKPOINT_NAME:-}" ] && EXTRA_OVERRIDES+=(paths.pretrain_checkpoint_name="$PRETRAIN_CHECKPOINT_NAME")

# ===== Find an available port =====
while command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 "$MASTER_PORT"; do
    MASTER_PORT=$((MASTER_PORT + 1))
done

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
# ===== Background monitor: copy logs and launch script into checkpoints =====
(
    while true; do
        sleep 60
        for ckpt in "$CHECKPOINT_DIR"/checkpoint-*; do
            [ -d "$ckpt" ] || continue
            if [ ! -f "${ckpt}/training_log.txt" ]; then
                cp "$LOG_FILE" "${ckpt}/training_log.txt" 2>/dev/null && \
                    cp "$0" "${ckpt}/launch_script.sh" 2>/dev/null && \
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saved log + script to ${ckpt}"
            fi
        done
    done
) > "$LOG_DIR/monitor_${NAME}.log" 2>&1 &
MONITOR_PID=$!
# Stop the background monitor on exit.
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# ===== Train =====
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="$MASTER_PORT" \
    -m latentskill.training.train_compiler \
    --config-name "$CONFIG_NAME" \
    mode="$MODE" \
    name="$NAME" \
    data.source="$SOURCE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.eval_batch_size="$EVAL_BATCH_SIZE" \
    run.use_gradient_checkpoint="$USE_GRADIENT_CHECKPOINT" \
    run.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    resume_global_step="$RESUME_GLOBAL_STEP" \
    optim.num_epochs="$NUM_EPOCHS" \
    optim.warmup_steps="$WARMUP_STEPS" \
    optim.learning_rate="$LEARNING_RATE" \
    eval.eval_steps="$EVAL_STEPS" \
    save.save_steps="$SAVE_STEPS" \
    data.context_max_length="$CONTEXT_MAX_LEN" \
    data.conversation_max_length="$CONVERSATION_MAX_LEN" \
    hypernetwork.type="$TYPE" \
    hypernetwork.transformer_cfg.num_layers="$NUM_LAYERS" \
    hypernetwork.method="$METHOD" \
    model.lora_r="$LORA_R" \
    model.metalora_r="$METALORA_R" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ===== Final log copy =====
LATEST=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    cp "$LOG_FILE" "${LATEST}/training_log_final.txt" 2>/dev/null
    cp "$0" "${LATEST}/launch_script.sh" 2>/dev/null
fi

exit $TRAIN_EXIT_CODE
