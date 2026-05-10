from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionInfo,
    SessionListResponse,
    SessionDetailResponse,
    MessageInfo,
    Source,
)
from backend.services.qa_service import qa_service
from backend.services.session_service import session_service

router = APIRouter(prefix="/api/qa", tags=["qa"])


# === V1 endpoints (unchanged) ===

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


# === V2: Streaming endpoint ===

@router.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    session_id = req.session_id or session_service.create_session()
    return StreamingResponse(
        qa_service.ask_stream(
            question=req.question,
            session_id=session_id,
            strategy=req.strategy,
            top_k=req.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


# === V2: Session management ===

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    sessions = session_service.list_sessions()
    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions]
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_service.get_messages(session_id)
    return SessionDetailResponse(
        id=session["id"],
        title=session["title"],
        messages=[
            MessageInfo(
                role=m["role"],
                content=m["content"],
                sources=[Source(**s) for s in m["sources"]] if m["sources"] else None,
                created_at=m["created_at"],
            )
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "deleted"}
