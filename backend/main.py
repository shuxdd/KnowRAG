import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import documents, qa
from backend.services.vector_service import get_document_count, get_embeddings
from backend.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: preload embeddings and vector store
    get_embeddings()
    get_document_count()
    yield


app = FastAPI(
    title="KnowRAG",
    version="0.1.0",
    description="Enterprise Knowledge Base RAG",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store_docs=get_document_count(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "error_code": "VALUE_ERROR"})
