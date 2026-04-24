from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import get_settings
from api import router

settings = get_settings()
app = FastAPI(title=settings.app_title, 
              description=settings.app_description,
              version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router, prefix="/v1", tags=["V1 API"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM API Service!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

Instrumentator().instrument(app).expose(app)