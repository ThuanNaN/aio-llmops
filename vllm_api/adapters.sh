#!/bin/bash
set -euo pipefail

PYTHON_CMD=$(command -v python3 || command -v python)
PORT=${VLLM_PORT:-8000}

download_snapshot() {
    local repo_id=$1
    "$PYTHON_CMD" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="${repo_id}"))
PY
}

load_adapter() {
    local alias=$1
    local repo_id=$2
    local adapter_path
    adapter_path=$(download_snapshot "$repo_id")

    echo "Loading adapter $alias from $repo_id"
    curl -fsS -X POST "http://localhost:${PORT}/v1/load_lora_adapter" \
        -H "Authorization: Bearer $VLLM_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"lora_name":"'"$alias"'","lora_path":"'"$adapter_path"'"}'
}

load_adapter "mathqa-lora" "${AIO_MATH_LORA_REPO:-VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora}"
load_adapter "vi-medqa-lora" "${AIO_MEDICAL_LORA_REPO:-VLAI-AIVN/Llama-3.2-1B-Instruct-vi-medqa-lora}"