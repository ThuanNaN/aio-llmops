import os
import time
import uuid
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

try:
    from tensorrt_llm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover - runtime dependency only
    LLM = None
    SamplingParams = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


REQUESTS = Counter(
    "trtllm_openai_requests_total",
    "Total OpenAI-compatible requests handled by the TensorRT-LLM gateway.",
    ["model"],
)
LATENCY = Histogram(
    "trtllm_openai_request_latency_seconds",
    "Latency for TensorRT-LLM OpenAI-compatible requests.",
    ["model"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False


def build_prompt(messages: list[ChatMessage]) -> str:
    rendered = [f"{message.role.upper()}: {message.content.strip()}" for message in messages if message.content.strip()]
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        first = result[0]
        outputs = getattr(first, "outputs", None)
        if outputs:
            generated = outputs[0]
            text = getattr(generated, "text", None)
            if text is not None:
                return text
        text = getattr(first, "text", None)
        if text is not None:
            return text
    return str(result)


class TensorRTLLMService:
    def __init__(self):
        self.model_name = os.getenv("TRTLLM_MODEL", "VLAI-AIVN/Llama-3.2-1B-Instruct-mathqa-lora")
        self.served_model_name = os.getenv("TRTLLM_SERVED_MODEL_NAME", "mathqa-lora")
        self.api_key = os.getenv("OPENAI_API_KEY", "aio2025")
        self.startup_error: str | None = None
        self.engine = None

        if IMPORT_ERROR is not None:
            self.startup_error = f"TensorRT-LLM import failed: {IMPORT_ERROR}"
            return

        try:
            self.engine = LLM(model=self.model_name)
        except Exception as exc:  # pragma: no cover - runtime dependency only
            self.startup_error = f"TensorRT-LLM initialization failed: {exc}"

    def ensure_ready(self):
        if self.engine is None:
            raise RuntimeError(self.startup_error or "TensorRT-LLM engine is not ready")

    def generate(self, messages: list[ChatMessage], temperature: float, max_tokens: int) -> str:
        self.ensure_ready()
        prompt = build_prompt(messages)
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        result = self.engine.generate(prompt, params)
        return extract_text(result).strip()


@lru_cache()
def get_service() -> TensorRTLLMService:
    return TensorRTLLMService()


def validate_authorization(authorization: str | None = Header(default=None)):
    expected = f"Bearer {get_service().api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(
    title="TensorRT-LLM OpenAI Gateway",
    description="OpenAI-compatible FastAPI facade for a TensorRT-LLM-served math model.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "TensorRT-LLM gateway ready"}


@app.get("/health")
def health_check():
    service = get_service()
    if service.startup_error:
        return {"status": "degraded", "detail": service.startup_error}
    return {"status": "healthy", "model": service.served_model_name}


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    validate_authorization(authorization)
    service = get_service()
    return {
        "object": "list",
        "data": [
            {
                "id": service.served_model_name,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionsRequest, authorization: str | None = Header(default=None)):
    validate_authorization(authorization)
    service = get_service()
    accepted_models = {service.served_model_name, service.model_name}
    if request.model and request.model not in accepted_models:
        raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented for this gateway")

    start_time = time.perf_counter()
    try:
        content = service.generate(request.messages, request.temperature, request.max_tokens)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency = time.perf_counter() - start_time
    model_name = request.model or service.served_model_name
    REQUESTS.labels(model_name).inc()
    LATENCY.labels(model_name).observe(latency)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)