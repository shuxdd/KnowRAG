"""
Pydantic 数据模型模块

本模块定义所有 API 请求和响应使用的数据模型：
- 问答相关: QuestionRequest, QuestionResponse, Source 等
- 文档相关: UploadResponse, DocumentInfo, DocumentListResponse 等
- 搜索相关: SearchRequest, SearchResponse, SearchResult 等
- 会话相关: SessionInfo, SessionListResponse, SessionDetailResponse 等
- 认证相关: AuthRegisterRequest, AuthLoginRequest, AuthTokenResponse 等
- 分块预览: ChunkPreviewResponse, ParentChunkPreview, LeafChunkPreview 等

所有模型继承自 Pydantic 的 BaseModel，用于：
- API 请求体验证
- API 响应序列化
- 自动生成 OpenAPI 文档
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    问答请求模型

    属性:
        question: 用户问题（1-2000字符）
        strategy: 检索策略（vector/hybrid/hybrid_rerank/fast/precise/deep/auto）
        top_k: 返回结果数量（1-50，默认为5）
        session_id: 会话 ID（可选，用于多轮对话）
    """
    question: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["vector", "hybrid", "hybrid_rerank", "fast", "precise", "deep", "auto"] = "auto"
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: Optional[str] = None


class Source(BaseModel):
    """
    文档来源模型

    表示检索结果中的一条文档来源。

    属性:
        content: 文档内容片段（前300字符）
        filename: 所属文件名
        score: 相似度分数
        heading_path: 标题路径
    """
    content: str
    filename: str
    score: float
    heading_path: list[str] | None = None


class QuestionResponse(BaseModel):
    """
    问答响应模型

    属性:
        answer: LLM 生成的回答
        sources: 检索到的文档来源列表
    """
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    """
    文档检索请求模型

    属性:
        query: 检索查询文本（1-2000字符）
        strategy: 检索策略
        top_k: 返回结果数量
    """
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["vector", "hybrid", "hybrid_rerank", "fast", "precise", "deep", "auto"] = "auto"
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    """
    单条检索结果模型

    属性:
        content: 文档内容
        filename: 文件名
        score: 相似度分数
    """
    content: str
    filename: str
    score: float


class SearchResponse(BaseModel):
    """检索响应模型，包含多条检索结果"""
    results: list[SearchResult]


class UploadResponse(BaseModel):
    """文档上传响应模型"""
    doc_id: str
    filename: str
    chunks_count: int


class DocumentInfo(BaseModel):
    """文档信息模型"""
    doc_id: str
    filename: str
    file_size: int
    chunks_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """文档列表响应模型"""
    documents: list[DocumentInfo]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    detail: str


class SessionInfo(BaseModel):
    """
    会话信息模型

    属性:
        id: 会话 ID
        title: 会话标题
        created_at: 创建时间
        updated_at: 最后更新时间
        message_count: 消息数量
    """
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    """会话列表响应模型"""
    sessions: list[SessionInfo]


class MessageInfo(BaseModel):
    """
    消息信息模型

    属性:
        role: 消息角色（user/assistant）
        content: 消息内容
        sources: 关联的来源列表
        created_at: 创建时间
    """
    role: str
    content: str
    sources: Optional[list[Source]] = None
    created_at: str


class SessionDetailResponse(BaseModel):
    """会话详情响应模型"""
    id: str
    title: str
    messages: list[MessageInfo]



class AuthRegisterRequest(BaseModel):
    """
    用户注册请求模型

    属性:
        username: 用户名（3-50字符，只能包含字母、数字、下划线）
        password: 密码（4-128字符）
    """
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=4, max_length=128)


class AuthLoginRequest(BaseModel):
    """用户登录请求模型"""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthUserResponse(BaseModel):
    """用户信息响应模型"""
    id: int
    username: str
    created_at: str


class AuthTokenResponse(BaseModel):
    """认证令牌响应模型"""
    access_token: str
    token_type: str = "bearer"


class LeafChunkPreview(BaseModel):
    """
    叶子块预览模型

    用于文档分块预览界面的叶子块信息。
    """
    chunk_index: int
    char_count: int
    preserve: bool
    undersized: bool
    content_preview: str


class ParentChunkPreview(BaseModel):
    """
    父块预览模型

    包含父块信息及其所有叶子块。
    """
    id: str
    heading_path: list[str]
    char_count: int
    page_start: int | None = None
    page_end: int | None = None
    created_at: str | None = None
    content_preview: str
    leaves: list[LeafChunkPreview]


class ChunkPreviewResponse(BaseModel):
    """分块预览响应模型"""
    filename: str
    parents: list[ParentChunkPreview]


class KGStatsResponse(BaseModel):
    entity_count: int
    relation_count: int
    type_count: int


class KGEntitySummary(BaseModel):
    id: str | None = None
    name: str
    type: str | None = None
    description: str | None = None


class KGRelationInfo(BaseModel):
    direction: str
    relation: str
    target: KGEntitySummary | None = None
    source: KGEntitySummary | None = None
    context: str | None = None


class KGChunkRef(BaseModel):
    chunk_id: str | None = None
    filename: str | None = None
    heading_path: str | None = None


class KGEntityDetailResponse(BaseModel):
    entity: KGEntitySummary
    relations: list[KGRelationInfo]
    mentioned_in: list[KGChunkRef]


class KGEntityListResponse(BaseModel):
    entities: list[KGEntitySummary]
    total: int
    page: int
    page_size: int
