from fastapi import APIRouter, HTTPException
from langsmith import traceable

from api.llm import get_gateway
from core.template import MATH_TEMPLATE
from models.math_qa import MathQARequest, MathQAResponse


router = APIRouter()


@traceable(name="math_qa_endpoint", run_type="chain", metadata={"route": "math_qa"})
@router.post("/math-qa", response_model=MathQAResponse)
def answer_math_question(request: MathQARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    prompt = MATH_TEMPLATE.format(question=request.question)
    result = get_gateway().route_chat(
        messages=[{"role": "user", "content": prompt}],
        requested_route="math_qa",
    )

    return MathQAResponse(
        content=result.content,
        route=result.route,
        provider=result.provider,
        model=result.model,
    )