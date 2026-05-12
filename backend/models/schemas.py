from pydantic import BaseModel, Field
from typing import Literal, Optional


# ==================== V1: 基础问答和数据模型 ====================

class QuestionRequest(BaseModel):
    """问答请求模型"""
    question: str = Field(..., min_length=1, max_length=2000)  # 问题内容，1-2000字符
    strategy: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"  # 检索策略
    top_k: int = Field(default=5, ge=1, le=50)  # 返回的文档数量，1-50
    session_id: Optional[str] = None  # 会话 ID（V2新增，可选）


class Source(BaseModel):
    """文档来源信息"""
    content: str    # 来源文档的内容摘要
    filename: str  # 来源文件名
    score: float   # 相关性分数
    heading_path: list[str] | None = None


class QuestionResponse(BaseModel):
    """问答响应模型"""
    answer: str              # LLM 生成的回答
    sources: list[Source]   # 答案引用的文档来源列表


class SearchRequest(BaseModel):
    """文档检索请求模型"""
    query: str = Field(..., min_length=1, max_length=2000)  # 查询文本
    strategy: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"  # 检索策略
    top_k: int = Field(default=5, ge=1, le=50)  # 返回结果数量


class SearchResult(BaseModel):
    """单个检索结果"""
    content: str   # 文档内容
    filename: str  # 文件名
    score: float   # 相关性分数


class SearchResponse(BaseModel):
    """检索响应模型"""
    results: list[SearchResult]  # 检索结果列表


class UploadResponse(BaseModel):
    """文档上传响应模型"""
    doc_id: str       # 文档 ID
    filename: str     # 文件名
    chunks_count: int # 分块数量


class DocumentInfo(BaseModel):
    """文档信息模型"""
    doc_id: str        # 文档 ID（实际为文件名）
    filename: str      # 文件名
    file_size: int     # 文件大小（字节）
    chunks_count: int  # 分块数量
    uploaded_at: str   # 上传时间（ISO 格式）


class DocumentListResponse(BaseModel):
    """文档列表响应模型"""
    documents: list[DocumentInfo]  # 文档列表


class ErrorResponse(BaseModel):
    """错误响应模型"""
    detail: str  # 错误详情


# ==================== V2: 会话管理模型 ====================

class SessionInfo(BaseModel):
    """会话基本信息"""
    id: str           # 会话 ID
    title: str        # 会话标题
    created_at: str   # 创建时间（ISO 格式）
    updated_at: str   # 最后更新时间（ISO 格式）
    message_count: int  # 消息数量


class SessionListResponse(BaseModel):
    """会话列表响应模型"""
    sessions: list[SessionInfo]  # 会话列表


class MessageInfo(BaseModel):
    """消息信息模型"""
    role: str                     # 消息角色（user/assistant）
    content: str                  # 消息内容
    sources: Optional[list[Source]] = None  # 关联的来源（可选）
    created_at: str               # 创建时间（ISO 格式）


class SessionDetailResponse(BaseModel):
    """会话详情响应模型"""
    id: str                      # 会话 ID
    title: str                   # 会话标题
    messages: list[MessageInfo]  # 消息列表


# ==================== V3: 评估模型 ====================

class EvalRunRequest(BaseModel):
    """评估运行请求模型"""
    strategy: Literal["all", "vector", "hybrid", "hybrid_rerank"] = "all"  # 评估策略


class EvalRunInfo(BaseModel):
    """评估运行信息模型"""
    id: str                      # 评估运行 ID
    strategy: str                # 使用的检索策略
    dataset_name: str            # 评估数据集名称
    question_count: int          # 问题数量
    started_at: str              # 开始时间（ISO 格式）
    completed_at: Optional[str] = None  # 完成时间（可选）
    avg_faithfulness: Optional[float] = None       # 平均忠实度
    avg_context_recall: Optional[float] = None    # 平均上下文召回率
    avg_context_precision: Optional[float] = None # 平均上下文精确率
    avg_answer_correctness: Optional[float] = None # 平均答案正确性
    avg_answer_accuracy: Optional[float] = None    # 平均答案准确率


class EvalResultItem(BaseModel):
    """单个评估结果项"""
    question: str                       # 问题
    ground_truth: str                  # 标准答案
    answer: str                        # 实际回答
    contexts: list[str]                # 检索到的上下文列表
    faithfulness: Optional[float] = None        # 忠实度
    context_recall: Optional[float] = None     # 上下文召回率
    context_precision: Optional[float] = None  # 上下文精确率
    answer_correctness: Optional[float] = None # 答案正确性
    answer_accuracy: Optional[float] = None    # 答案准确率


class EvalRunDetail(BaseModel):
    """评估运行详情模型"""
    id: str                      # 评估运行 ID
    strategy: str                # 检索策略
    dataset_name: str            # 数据集名称
    started_at: str              # 开始时间
    completed_at: Optional[str] = None  # 完成时间
    avg_faithfulness: Optional[float] = None       # 平均忠实度
    avg_context_recall: Optional[float] = None    # 平均召回率
    avg_context_precision: Optional[float] = None # 平均精确率
    avg_answer_correctness: Optional[float] = None # 平均正确性
    avg_answer_accuracy: Optional[float] = None    # 平均准确率
    results: list[EvalResultItem]  # 所有问题的评估结果


class EvalListResponse(BaseModel):
    """评估列表响应模型"""
    runs: list[EvalRunInfo]  # 评估运行列表
