from pydantic import BaseModel


class MedicalQARequest(BaseModel):
    question: str
    context: str | None = None


class MedicalQAResponse(BaseModel):
    content: str
    route: str
    provider: str
    model: str