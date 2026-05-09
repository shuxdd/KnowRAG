from fastapi import APIRouter
from backend.models.schemas import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from backend.services.qa_service import qa_service

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    result = qa_service.ask(
        question=req.question,
        strategy=req.strategy,
        top_k=req.top_k,
    )
    return QuestionResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@router.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    docs = qa_service.search(
        query=req.query,
        strategy=req.strategy,
        top_k=req.top_k,
    )
    results = [
        SearchResult(
            content=doc.page_content,
            filename=doc.metadata.get("filename", "unknown"),
            score=doc.metadata.get("score", 0.0),
        )
        for doc in docs
    ]
    return SearchResponse(results=results)
