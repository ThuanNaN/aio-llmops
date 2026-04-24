from fastapi import APIRouter, HTTPException
from langsmith import traceable

from api.llm import get_gateway
from core.template import MEDICAL_TEMPLATE
from models.medical_qa import MedicalQARequest, MedicalQAResponse

router = APIRouter()


@traceable(name="medical_qa_endpoint", run_type="chain", metadata={"route": "medical_qa"})
@router.post("/medical-qa", response_model=MedicalQAResponse)
def answer_medical_question(request: MedicalQARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    prompt = MEDICAL_TEMPLATE.format(
        question=request.question,
        context=request.context.strip() if request.context else "No additional context provided.",
    )
    result = get_gateway().route_chat(
        messages=[{"role": "user", "content": prompt}],
        requested_route="medical_qa",
    )

    return MedicalQAResponse(
        content=result.content,
        route=result.route,
        provider=result.provider,
        model=result.model,
    )
