# LLM Backend Service

A FastAPI gateway that routes math and medical requests across multi-LoRA vLLM and TensorRT-LLM backends using LangChain.

The medical QA flow is aligned to the Vietnamese free-form QA dataset `hungnm/vietnamese-medical-qa`, so it accepts open-ended questions instead of answer choices.

The default config now targets local ports so the repo can run on one host with vLLM on `8000` and TensorRT-LLM on `8002`, or on separate hosts by changing `VLLM_HOST` and `TRTLLM_HOST`.

## Features

- **LLM Classification**: An LLM classifier selects the best route for each incoming chat request
- **Math QA**: Routes math workloads to TensorRT-LLM by default
- **Medical QA**: Routes medical workloads to multi-LoRA vLLM
- **Shared Exact-Match Cache**: Redis-backed caching for classifier decisions and repeated routed responses
- **Route Overrides**: Supports explicit route selection for backend comparison and debugging
- **Instrumentation**: Gateway counters and latency histograms exposed to Prometheus
- **Tracing and Evaluation**: LangSmith integration for request traces and evaluation datasets

## Architecture

The service follows a clean, modular architecture:

- **API Layer**: FastAPI routes for each capability
- **LLM Integration**: LangChain for LLM orchestration and route classification
- **Routing Config**: YAML-driven route and provider selection
- **Caching**: Shared Redis cache for exact-match classifier and response reuse
- **Prompting**: Structured templates for consistent model outputs
- **Observability**: LangSmith traces plus Prometheus metrics

## API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- `POST /v1/chat`: Routed chat endpoint with optional route override
- `POST /v1/math-qa`: Force the math route
- `POST /v1/medical-qa`: Answer free-form Vietnamese medical questions with optional context
- `GET /metrics`: Prometheus metrics endpoint

## Environment Variables

The service is configured using the following environment variables:

- `OPENAI_API_KEY`: API key for the LLM service
- `VLLM_HOST`: Hostname or IP address for the vLLM API
- `VLLM_PORT`: Port exposed by the vLLM API
- `TRTLLM_HOST`: Hostname or IP address for the TensorRT-LLM API
- `TRTLLM_PORT`: Port exposed by the TensorRT-LLM API
- `BACKEND_PORT`: Port exposed by the FastAPI gateway itself
- `ROUTING_CONFIG_PATH`: Path to the YAML routing config
- `REQUEST_TIMEOUT_SECONDS`: Timeout used for upstream model requests
- `CACHE_ENABLED`: Enable or disable the gateway cache
- `CACHE_REDIS_URL`: Redis connection string used for shared cache entries
- `CACHE_NAMESPACE`: Manual cache-busting namespace for model or adapter rollouts
- `CLASSIFIER_CACHE_ENABLED`: Cache classifier decisions when enabled
- `RESPONSE_CACHE_ENABLED`: Cache final routed responses when enabled
- `CACHE_SOCKET_TIMEOUT_SECONDS`: Redis socket timeout for cache operations
- `LANGSMITH_TRACING`: Enable/disable LangSmith tracing
- `LANGSMITH_API_KEY`: LangSmith API key for tracing
- `LANGSMITH_PROJECT`: LangSmith project name

## Running the Service

### Using Docker Compose

```bash
docker compose up -d
```

This starts both the FastAPI gateway and a local Redis cache. The API will be available at `http://localhost:${BACKEND_PORT}`.

### Development Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`

3. Run the development server:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

## Cache Operations

The gateway uses exact-match Redis caching in two places:

- Classifier decisions for repeated `/v1/chat` conversations
- Final routed responses for repeated prompts and forced-route requests

For multi-replica deployments, point every backend replica at the same Redis instance using `CACHE_REDIS_URL`.

When LoRA weights, provider deployments, or model contents change without a route name change, bump `CACHE_NAMESPACE` and restart the gateway to invalidate old cache entries.

Prometheus exposes cache counters through `llm_gateway_cache_operations_total` with `cache=classifier|response` and `result=hit|miss|store|error`.

## Evaluation

Run the LangSmith routing evaluation script after the gateway is up:

```bash
python evals/eval_sa.py
```
