# AIO LLM Ops

An end-to-end LLMOps stack for routed math and medical question answering with multi-LoRA vLLM, TensorRT-LLM, LangChain routing, LangSmith tracing, and Prometheus/Grafana/Loki/Promtail observability.

## Architecture

- **vLLM API**: Serves the base `meta-llama/Llama-3.2-1B-Instruct` model as `routing-classifier` and dynamically loads two LoRA adapters:
  - `VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora`
   - `VLAI-AIVN/Llama-3.2-1B-Instruct-vi-medqa-lora` for free-form Vietnamese medical QA aligned to `hungnm/vietnamese-medical-qa`
- **TensorRT-LLM API**: Exposes an OpenAI-compatible chat completions endpoint for `VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora`
- **FastAPI Gateway**: Uses LangChain plus an LLM classifier to route requests to the best backend, caches exact-match classifier and response results in Redis, and records gateway metrics and LangSmith traces
- **Frontend**: Gradio interface with routed chat, math QA, and free-form medical QA tabs
- **Monitoring**: Prometheus for metrics, Grafana for dashboards, Loki for logs, and Promtail for Docker log collection

## Deployment Layout

- **Single host default**: vLLM on `GPU 0` at port `8000`, TensorRT-LLM on `GPU 1` at port `8002`, plus backend, frontend, and monitoring on the same machine
- **Split host option**: Point the backend and monitoring stack at any `VLLM_HOST` and `TRTLLM_HOST` values, such as `PC1` (`3090`, `24 GB`) for vLLM plus router and `PC2` (`3060`, `12 GB`) for TensorRT-LLM

## Gateway Endpoints

- `POST /v1/chat`: Routed chat with optional route override (`math_qa`, `medical_qa`)
- `POST /v1/math-qa`: Force the TensorRT-LLM math route
- `POST /v1/medical-qa`: Force the vLLM medical route
- `GET /metrics`: Gateway Prometheus metrics

## Getting Started

1. Create the shared Docker network:

   ```bash
   docker network create aio-network
   ```

2. Copy the example env files and fill in secrets where needed:

   ```bash
   cp .env.example .env
   cp backend/.env.example backend/.env
   cp vllm_api/.env.example vllm_api/.env
   cp trtllm_api/.env.example trtllm_api/.env
   cp frontend/.env.example frontend/.env
   cp monitor/.env.example monitor/.env
   ```

3. Choose the topology in the env files:

   Single host example:

   ```bash
   # vllm_api/.env
   VLLM_PORT="8000"
   VLLM_DEVICE_ID="0"

   # trtllm_api/.env
   TRTLLM_PORT="8002"
   TRTLLM_DEVICE_ID="1"

   # backend/.env and monitor/.env
   VLLM_HOST="127.0.0.1"
   VLLM_PORT="8000"
   TRTLLM_HOST="127.0.0.1"
   TRTLLM_PORT="8002"
   BACKEND_PORT="8001"
   ```

   Split host example:

   ```bash
   # PC1: vllm_api/.env
   VLLM_PORT="8000"
   VLLM_DEVICE_ID="0"

   # PC2: trtllm_api/.env
   TRTLLM_PORT="8000"
   TRTLLM_DEVICE_ID="0"

   # PC1: backend/.env and monitor/.env
   VLLM_HOST="192.168.1.101"
   VLLM_PORT="8000"
   TRTLLM_HOST="192.168.1.102"
   TRTLLM_PORT="8000"
   BACKEND_PORT="8001"
   ```

4. Start the services on each machine:

   For a single host:

   ```bash
   ./run.sh up
   ```

   For a split deployment, on the vLLM and router node:

   ```bash
   ./run.sh up-vllm
   ./run.sh up-app
   ```

   On the TensorRT-LLM node:

   ```bash
   ./run.sh up-trtllm
   ```

## Accessing Services

- **vLLM API**: `http://$VLLM_HOST:$VLLM_PORT`
- **TensorRT-LLM API**: `http://$TRTLLM_HOST:$TRTLLM_PORT`
- **FastAPI Gateway**: `http://$BACKEND_HOST:$BACKEND_PORT`
- **Gradio UI**: `http://$BACKEND_HOST:$FRONTEND_PORT`
- **Open WebUI**: `http://$BACKEND_HOST:$OPEN_WEBUI_PORT`
- **Grafana**: `http://$BACKEND_HOST:3000`
- **Prometheus**: `http://$BACKEND_HOST:9090`
- **Loki**: `http://$BACKEND_HOST:3100`

## Route Overrides

- `math_qa`: TensorRT-LLM math route
- `medical_qa`: vLLM medical LoRA route

## Cache Strategy

- vLLM prefix caching stays enabled for provider-side KV reuse
- The backend adds an exact-match Redis cache for classifier decisions and repeated responses
- For model or LoRA rollouts, bump `CACHE_NAMESPACE` on every backend replica to invalidate old entries without changing route names

## Benchmark

```bash
export OPENAI_API_KEY=<your gateway api key>
make bench_serving
```
