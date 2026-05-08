from fastapi import APIRouter, HTTPException

from backend.services.qa_service import (
    ask_question,
    get_session_history_messages,
    clear_session_history,
)
from backend.models.schemas import QuestionRequest, AnswerResponse, SessionHistory

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    try:
        return await ask_question(req.question, req.top_k, req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@router.get("/history/{session_id}", response_model=SessionHistory)
async def get_history(session_id: str):
    return SessionHistory(
        session_id=session_id,
        messages=get_session_history_messages(session_id),
    )


@router.delete("/history/{session_id}")
async def delete_history(session_id: str):
    clear_session_history(session_id)
    return {"detail": "ok", "session_id": session_id}
