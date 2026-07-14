#!/usr/bin/env bash
# run_eval.sh
# Start the retrieval server, wait until it is ready, then run SearchQA evaluation.
# Run from the project root:
#   bash evals/searchqa/run_eval.sh <checkpoint-dir> [port]
#
# Example:
#   bash evals/searchqa/run_eval.sh \
#       <checkpoint-dir>

set -euo pipefail

CHECKPOINT=${1:-${CHECKPOINT:-}}
PORT=${2:-${PORT:-8000}}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash evals/searchqa/run_eval.sh <checkpoint-dir> [port]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WIKI_INDEX=${WIKI_INDEX:-"${PROJECT_ROOT}/wiki_index/e5_Flat.index"}
WIKI_CORPUS=${WIKI_CORPUS:-"${PROJECT_ROOT}/wiki_index/wiki-18.jsonl"}
E5_MODEL=${E5_MODEL:-"intfloat/e5-base-v2"}
TEST_DATA=${TEST_DATA:-"${PROJECT_ROOT}/data/search_test/search_test_all.jsonl"}
SKILL_CONTEXT_DIR=${SKILL_CONTEXT_DIR:-"${PROJECT_ROOT}/evals/searchqa/skills"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/evals/searchqa/results"}
RETRIEVAL_SERVER="${PROJECT_ROOT}/evals/searchqa/retrieval_server.py"

RETRIEVAL_URL="http://127.0.0.1:${PORT}/retrieve"
LOG_DIR="${PROJECT_ROOT}/evals/searchqa/logs"
mkdir -p "$LOG_DIR"
RETRIEVAL_LOG="${LOG_DIR}/retrieval_server.log"
STARTED_RETRIEVAL=0

cleanup() {
    if [ "${STARTED_RETRIEVAL:-0}" = "1" ]; then
        kill "$RETRIEVAL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DEVICE=${DEVICE:-cuda}

echo "=========================================="
echo "LatentSkill Search Benchmark Evaluation"
echo "Checkpoint : ${CHECKPOINT}"
echo "Port       : ${PORT}"
echo "=========================================="

if [ ! -f "$WIKI_INDEX" ]; then
    echo "[error] index file not found: $WIKI_INDEX"
    exit 1
fi
if [ ! -f "$WIKI_CORPUS" ]; then
    echo "[error] corpus file not found: $WIKI_CORPUS"
    exit 1
fi

if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "[warning] port ${PORT} is already in use; assuming retrieval server is running."
else
    echo "[1/3] Starting retrieval server in the background. Log: ${RETRIEVAL_LOG}"
    python "$RETRIEVAL_SERVER" \
        --index_path  "$WIKI_INDEX" \
        --corpus_path "$WIKI_CORPUS" \
        --retriever_name e5 \
        --retriever_model "$E5_MODEL" \
        --topk 3 \
        --port "$PORT" \
        > "$RETRIEVAL_LOG" 2>&1 &
    RETRIEVAL_PID=$!
    STARTED_RETRIEVAL=1
    echo "  Retrieval server PID: ${RETRIEVAL_PID}"

    echo "[2/3] Waiting for retrieval server readiness..."
    MAX_WAIT=600
    WAITED=0
    until nc -z 127.0.0.1 "$PORT" 2>/dev/null; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "[error] retrieval server was not ready within ${MAX_WAIT}s. Check ${RETRIEVAL_LOG}"
            exit 1
        fi
        echo "  waited ${WAITED}s..."
    done
    echo "  [OK] retrieval server ready (${WAITED}s)"
fi

echo "[3/3] Running LatentSkill SearchQA evaluation..."
cd "$PROJECT_ROOT"

python -m evals.searchqa.evaluate \
    --checkpoint        "$CHECKPOINT" \
    --config_name       models/qwen3_8b \
    --test_data         "$TEST_DATA" \
    --skill_context_dir "$SKILL_CONTEXT_DIR" \
    --retrieval_url     "$RETRIEVAL_URL" \
    --retrieval_topk    3 \
    --max_steps         4 \
    --max_new_tokens    4096 \
    --context_max_length    4096 \
    --conversation_max_length 4096 \
    --output_dir        "$OUTPUT_DIR" \
    --device            "$DEVICE"

echo "=========================================="
echo "Evaluation complete. Results: ${OUTPUT_DIR}"
echo "=========================================="
