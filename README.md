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

- **192.168.1.101**: vLLM serving node with `24 GB` VRAM, plus the default backend, frontend, and monitoring stack in this repo configuration
- **192.168.1.102**: TensorRT-LLM serving node for the math route

## Gateway Endpoints

- `POST /v1/chat`: Routed chat with optional route override (`math_qa`, `math_qa_vllm`, `medical_qa`)
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
   ```

3. Start the services on each machine:

   On `192.168.1.101`:

   ```bash
   ./run.sh up-vllm
   ./run.sh up-app
   ```

   On `192.168.1.102`:

   ```bash
   ./run.sh up-trtllm
   ```

4. For single-machine development only, you can still start everything together:

   ```bash
   ./run.sh up
   ```

## Accessing Services

- **vLLM API**: `http://192.168.1.101:8000`
- **TensorRT-LLM API**: `http://192.168.1.102:8000`
- **FastAPI Gateway**: `http://192.168.1.101:8001`
- **Gradio UI**: `http://192.168.1.101:7860`
- **Open WebUI**: `http://192.168.1.101:8080`
- **Grafana**: `http://192.168.1.101:3000`
- **Prometheus**: `http://192.168.1.101:9090`
- **Loki**: `http://192.168.1.101:3100`

## Route Overrides

- `math_qa`: TensorRT-LLM math route
- `math_qa_vllm`: vLLM math LoRA route for comparison/debugging
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
