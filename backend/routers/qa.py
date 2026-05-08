import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.services.qa_service import (
    ask_question,
    ask_question_stream,
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


@router.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    async def event_stream():
        try:
            async for event in ask_question_stream(req.question, req.top_k, req.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
