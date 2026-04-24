# TensorRT-LLM Serving Layer

This service exposes an OpenAI-compatible chat completions API backed by TensorRT-LLM for the math LoRA model.

In the default distributed deployment, this service runs on `192.168.1.102` and is exposed on port `8000`.

## Model

- `VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora`

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /metrics`

## Notes

- The container expects GPU access.
- The TensorRT-LLM Python runtime is installed during image build.
- The FastAPI gateway routes math workloads to this service by default.