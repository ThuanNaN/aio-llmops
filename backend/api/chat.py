from fastapi import APIRouter, HTTPException
from langsmith import traceable

from api.llm import get_gateway
from models.chat import ChatRequest, ChatResponse


router = APIRouter()


@traceable(name="chat_gateway_endpoint", run_type="chain")
@router.post("/chat", response_model=ChatResponse)
def route_chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one chat message is required.")

    result = get_gateway().route_chat(
        messages=[message.model_dump() for message in request.messages],
        requested_route=request.route,
    )
    return ChatResponse(
        content=result.content,
        route=result.route,
        provider=result.provider,
        model=result.model,
        classifier_model=result.classifier_model,
        reason=result.reason,
    )