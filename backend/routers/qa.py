from fastapi import APIRouter, HTTPException

from backend.services.qa_service import ask_question
from backend.models.schemas import QuestionRequest, AnswerResponse

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    try:
        return await ask_question(req.question, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")
