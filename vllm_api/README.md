# Serving LLMs with vLLM

This component provides multi-LoRA inference for Llama 3.2 1B using vLLM and exposes the base model as the gateway classifier.

By default, this service runs on `localhost` and can be pinned to any GPU with `AIO_DEVICE_ID`.

## Features

- **Base Model**: Serves `meta-llama/Llama-3.2-1B-Instruct` as `routing-classifier`
- **LoRA Adapters**: Dynamically downloads and loads math and medical LoRA adapters from Hugging Face
- **OpenAI-compatible API**: Compatible with standard OpenAI client libraries
- **High Performance**: Optimized inference with continuous batching
- **Metrics**: Prometheus integration for comprehensive monitoring

## Architecture

The service uses vLLM to provide efficient inference with:

- Optimized GPU utilization
- Continuous batching for high throughput
- PagedAttention for memory efficiency
- Dynamic LoRA adapter switching

## LoRA Adapters

The service loads the following adapters at startup:

1. **MathQA Adapter**: `VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora`
2. **VI-Medical-QA Adapter**: `VLAI-AIVN/Llama-3.2-1B-Instruct-vi-medqa-lora`, used for free-form Vietnamese medical QA aligned to `hungnm/vietnamese-medical-qa`

## Environment Variables

Configure the service with:

- `VLLM_API_KEY`: API key for authorization
- `AIO_BASE_MODEL`: Base model served by vLLM
- `AIO_SERVED_MODEL_NAME`: Alias exposed for the classifier model
- `AIO_MATH_LORA_REPO`: Hugging Face repository for the math LoRA adapter
- `AIO_MEDICAL_LORA_REPO`: Hugging Face repository for the medical LoRA adapter
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING`: Enable/disable runtime LoRA updates
- `VLLM_PORT`: Port exposed by the API server
- `AIO_DEVICE_ID`: GPU device index to expose to the container

## Running the Service

### Using Docker Compose

```bash
docker compose up -d
```

The vLLM API will be available at `http://localhost:8000`.

For a split deployment, set `VLLM_PORT` and the backend's `VLLM_HOST` and `VLLM_PORT` to match the host where this service runs.

### Hardware Requirements

- NVIDIA GPU with at least 12GB VRAM
- CUDA 13.0
- At least 16GB system RAM
