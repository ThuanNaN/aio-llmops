# TensorRT-LLM Serving Layer

This service exposes an OpenAI-compatible chat completions API backed by TensorRT-LLM for the math LoRA model.

By default, this service runs on `localhost` and is exposed on port `8002` so it can coexist with vLLM on the same machine.

## Model

- `VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora`

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /metrics`

## Notes

- The container expects GPU access.
- Use `TRTLLM_DEVICE_ID` to choose which GPU is exposed to the container.
- Use `TRTLLM_PORT` to change the listening port for single-host or multi-host layouts.
- The TensorRT-LLM Python runtime is installed during image build.
- The FastAPI gateway routes math workloads to this service by default.