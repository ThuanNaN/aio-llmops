from pydantic import BaseModel


class MathQARequest(BaseModel):
    question: str


class MathQAResponse(BaseModel):
    content: str
    route: str
    provider: str
    model: str