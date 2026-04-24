#!/bin/bash
set -euo pipefail

PYTHON_CMD=$(command -v python3 || command -v python)

if [ -z "$PYTHON_CMD" ]; then
  echo "Python executable not found."
  exit 1
fi

BASE_MODEL=${VLLM_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-routing-classifier}
PORT=${VLLM_PORT:-8000}

echo "Starting vLLM with base model: $BASE_MODEL"

"$PYTHON_CMD" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --api-key "$VLLM_API_KEY" \
  --enable-lora \
  --max-lora-rank "${VLLM_MAX_LORA_RANK:-64}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --swap-space "${VLLM_SWAP_SPACE:-16}" \
  --disable-log-requests \
  --enable-prefix-caching &

VLLM_PID=$!

wait_for_api() {
  echo "Waiting for vLLM API to become ready..."
  until curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "http://localhost:${PORT}/v1/models" >/dev/null; do
    sleep 5
  done
}

wait_for_api

echo "Loading LoRA adapters into vLLM..."
bash /app/adapters.sh

wait "$VLLM_PID"
