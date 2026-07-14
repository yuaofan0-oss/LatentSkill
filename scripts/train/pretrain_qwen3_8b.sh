#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

NAME=${NAME:-latentskill_pretrain_qwen3_8b}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_NAME=${CONFIG_NAME:-models/qwen3_8b}
SOURCE=${SOURCE:-skill-pretrain}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
USE_GRADIENT_CHECKPOINT=${USE_GRADIENT_CHECKPOINT:-true}
RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:--1}

LEARNING_RATE=${LEARNING_RATE:-5e-5}
CONVERSATION_MAX_LEN=${CONVERSATION_MAX_LEN:-4096}
CONTEXT_MAX_LEN=${CONTEXT_MAX_LEN:-$((CONVERSATION_MAX_LEN - 9))}
TYPE=${TYPE:-transformer}
NUM_LAYERS=${NUM_LAYERS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-200}
METHOD=${METHOD:-rl}
LORA_R=${LORA_R:-8}
METALORA_R=${METALORA_R:-128}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-200}

case "$SOURCE" in
    skill-pretrain|skill-pretrain-dynamic) ;;
    *)
        echo "Unsupported SOURCE='$SOURCE'. Use skill-pretrain or skill-pretrain-dynamic." >&2
        exit 2
        ;;
esac

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

LOG_DIR=${LOG_DIR:-logs}
LOG_FILE="$LOG_DIR/pretrain_$NAME.log"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$NAME/pretrain"
mkdir -p "$LOG_DIR"

while command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 "$MASTER_PORT"; do
    MASTER_PORT=$((MASTER_PORT + 1))
done

EXTRA_OVERRIDES=()
[ -n "${MODEL_PATH:-}" ] && EXTRA_OVERRIDES+=(paths.model_path="$MODEL_PATH")
[ -n "${DATA_ROOT:-}" ] && EXTRA_OVERRIDES+=(data.root_dir="$DATA_ROOT")
[ -n "${CHECKPOINT_ROOT:-}" ] && EXTRA_OVERRIDES+=(paths.checkpoint_root="$CHECKPOINT_ROOT")

# Step 1: build static skill group indices when needed.
python3 -m latentskill.data.group_index \
    --config-name "$CONFIG_NAME" \
    name="$NAME" \
    mode=pretrain \
    data.source="$SOURCE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.eval_batch_size="$EVAL_BATCH_SIZE" \
    run.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    run.use_gradient_checkpoint="$USE_GRADIENT_CHECKPOINT" \
    resume_global_step="$RESUME_GLOBAL_STEP" \
    optim.learning_rate="$LEARNING_RATE" \
    hypernetwork.type="$TYPE" \
    data.conversation_max_length="$CONVERSATION_MAX_LEN" \
    data.context_max_length="$CONTEXT_MAX_LEN" \
    hypernetwork.transformer_cfg.num_layers="$NUM_LAYERS" \
    optim.warmup_steps="$WARMUP_STEPS" \
    hypernetwork.method="$METHOD" \
    save.save_steps="$SAVE_STEPS" \
    eval.eval_steps="$EVAL_STEPS" \
    model.lora_r="$LORA_R" \
    model.metalora_r="$METALORA_R" \
    "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$LOG_FILE"

# Check whether group-index generation succeeded.
GEN_EXIT_CODE=${PIPESTATUS[0]}
if [ $GEN_EXIT_CODE -ne 0 ]; then
    echo "latentskill.data.group_index failed with exit code $GEN_EXIT_CODE"
    exit $GEN_EXIT_CODE
fi

# Start background checkpoint log archiving.
mkdir -p "$CHECKPOINT_DIR"
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

# Step 2: train.
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="$MASTER_PORT" \
    -m latentskill.training.train_compiler \
    --config-name "$CONFIG_NAME" \
    name="$NAME" \
    mode=pretrain \
    data.source="$SOURCE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.eval_batch_size="$EVAL_BATCH_SIZE" \
    run.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    run.use_gradient_checkpoint="$USE_GRADIENT_CHECKPOINT" \
    resume_global_step="$RESUME_GLOBAL_STEP" \
    optim.learning_rate="$LEARNING_RATE" \
    hypernetwork.type="$TYPE" \
    data.conversation_max_length="$CONVERSATION_MAX_LEN" \
    data.context_max_length="$CONTEXT_MAX_LEN" \
    hypernetwork.transformer_cfg.num_layers="$NUM_LAYERS" \
    optim.warmup_steps="$WARMUP_STEPS" \
    hypernetwork.method="$METHOD" \
    save.save_steps="$SAVE_STEPS" \
    eval.eval_steps="$EVAL_STEPS" \
    model.lora_r="$LORA_R" \
    model.metalora_r="$METALORA_R" \
    "${EXTRA_OVERRIDES[@]}" 2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Copy the final log and launch script after training ends.
LATEST=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    cp "$LOG_FILE" "${LATEST}/training_log_final.txt" 2>/dev/null
    cp "$0" "${LATEST}/launch_script.sh" 2>/dev/null
fi

exit $TRAIN_EXIT_CODE
