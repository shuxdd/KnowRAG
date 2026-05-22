"""
问答路由模块

提供问答检索和会话管理接口。

接口列表：
- POST /api/qa/ask: 非流式问答（V1）
- POST /api/qa/search: 文档检索（V1）
- POST /api/qa/ask/stream: 流式问答（V2）
- POST /api/qa/agent: Agent 流式问答（V3）
- GET /api/qa/sessions: 获取会话列表（V2）
- GET /api/qa/sessions/{session_id}: 获取会话详情（V2）
- DELETE /api/qa/sessions/{session_id}: 删除会话（V2）

检索策略：
- fast: 快速检索，仅使用向量检索
- precise: 精确检索，向量 + BM25 混合
- deep: 深度检索，向量 + BM25 + HyDE + Rerank
- auto: 自动选择策略（根据问题复杂度）
- hybrid/hybrid_rerank: 与 precise/deep 等价

会话管理：
- 支持多轮对话
- 自动保存对话历史
- 流式返回答案（SSE 格式）
"""

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
    AgentRequest,
)
from backend.services.qa_service import qa_service
from backend.services.session_service import session_service
from backend.services.agent_service import agent_service

router = APIRouter(prefix="/api/qa", tags=["qa"])


# === V1 接口（基础问答） ===

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """
    非流式问答接口（V1）

    Args:
        req: 包含问题、检索策略等信息的请求体

    Returns:
        包含答案和来源信息的响应
    """
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
    """
    文档检索接口（V1）

    Args:
        req: 包含查询文本和检索策略的请求体

    Returns:
        检索结果列表，包含文档内容、文件名和相关性分数
    """
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


# === V2: 流式问答接口 ===

@router.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    """
    流式问答接口（V2）
    支持 SSE（Server-Sent Events）流式输出和会话管理

    Args:
        req: 包含问题、检索策略、会话 ID 等信息的请求体

    Returns:
        StreamingResponse，SSE 格式的数据流
        - sources: 检索来源信息（首包）
        - token: LLM 生成的文本片段
        - done: 结束标识
    """
    # 如果没有提供 session_id，自动创建一个新会话
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
            "X-Session-Id": session_id,  # 返回新创建的会话 ID
        },
    )


# === V3: Agent 流式问答接口 ===

@router.post("/agent")
async def ask_agent(req: AgentRequest):
    """
    Agent 流式问答接口 (V3)
    使用 LangGraph Agent 自主决定检索策略和工具调用

    SSE 事件: tool → token → sources → error (如发生) → done
    """
    session_id = req.session_id or session_service.create_session()
    history = session_service.get_history(session_id)
    return StreamingResponse(
        agent_service.ask_stream(
            question=req.question,
            session_id=session_id,
            chat_history_messages=history.messages if history else None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


# === V2: 会话管理接口 ===

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    获取会话列表接口（V2）

    Returns:
        所有会话的列表，按最后更新时间降序排列
    """
    sessions = session_service.list_sessions()
    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions]
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """
    获取指定会话的详情（V2）

    Args:
        session_id: 会话 ID

    Returns:
        会话详情，包含会话信息和所有消息列表

    Raises:
        HTTPException: 如果会话不存在，返回 404 错误
    """
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
    """
    删除指定会话（V2）

    Args:
        session_id: 要删除的会话 ID

    Returns:
        删除成功的确认信息

    Raises:
        HTTPException: 如果会话不存在，返回 404 错误
    """
    if not session_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "deleted"}
