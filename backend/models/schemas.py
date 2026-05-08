from pydantic import BaseModel, Field
from datetime import datetime


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: datetime
    size_bytes: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    session_id: str = Field(default="default")


class SourceCitation(BaseModel):
    doc_id: str
    filename: str
    content_snippet: str
    chunk_index: int = 0
    page: int | None = None


class HistoryMessage(BaseModel):
    role: str  # "human" | "ai"
    content: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]
    session_id: str = "default"


class SessionHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_docs: int


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
