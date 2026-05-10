import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import documents, qa, eval

app = FastAPI(
    title="KnowRAG - Enterprise Knowledge Base",
    version="1.0.0",
    description="RAG-powered enterprise knowledge base with hybrid search and reranking",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(eval.router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
