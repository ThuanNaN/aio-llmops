#!/bin/bash
set -euo pipefail

PYTHON_CMD=$(command -v python3 || command -v python)

if [ -z "$PYTHON_CMD" ]; then
  echo "Python executable not found."
  exit 1
fi

BASE_MODEL=${AIO_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
SERVED_MODEL_NAME=${AIO_SERVED_MODEL_NAME:-routing-classifier}
PORT=${VLLM_PORT:-8000}

echo "Starting vLLM with base model: $BASE_MODEL"

"$PYTHON_CMD" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --api-key "$VLLM_API_KEY" \
  --enable-lora \
  --max-lora-rank 16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-loras 8 \
  --max-cpu-loras 16 &

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
