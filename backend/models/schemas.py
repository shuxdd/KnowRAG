from pydantic import BaseModel, Field
from typing import Literal, Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: Optional[str] = None


class Source(BaseModel):
    content: str
    filename: str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    content: str
    filename: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_size: int
    chunks_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class ErrorResponse(BaseModel):
    detail: str


# === V2: Session models ===

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


class MessageInfo(BaseModel):
    role: str
    content: str
    sources: Optional[list[Source]] = None
    created_at: str


class SessionDetailResponse(BaseModel):
    id: str
    title: str
    messages: list[MessageInfo]
