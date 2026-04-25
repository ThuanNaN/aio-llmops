from fastapi import APIRouter
from .chat import router as chat_router
from .math_qa import router as math_qa_router
from .medical_qa import router as medical_qa_router

router = APIRouter()
router.include_router(chat_router, tags=["Routed Chat"])
router.include_router(math_qa_router, tags=["Math QA"])
router.include_router(medical_qa_router, tags=["Medical QA"])