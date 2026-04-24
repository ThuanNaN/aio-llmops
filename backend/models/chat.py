from typing import List, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    route: str | None = None


class ChatResponse(BaseModel):
    content: str
    route: str
    provider: str
    model: str
    classifier_model: str
    reason: str