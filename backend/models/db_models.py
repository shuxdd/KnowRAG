"""
数据库 ORM 模型模块

本模块定义与 PostgreSQL 数据库表映射的 SQLAlchemy ORM 模型：
- ParentChunkORM: 父块（Parent Chunk）表，存储文档的父级分块
- UserORM: 用户表，存储用户账号信息

ORM 模型用于与数据库交互，进行 CRUD 操作。
"""

from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Integer, Float, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy 声明性基类，所有 ORM 模型都继承此类"""
    pass


class ParentChunkORM(Base):
    """
    父块 ORM 模型

    存储文档的父级分块信息，每个父块代表文档中的一个逻辑章节。
    父块由多个叶子块（Leaf Chunk）组成，用于检索时返回完整的上下文。

    属性:
        id: 唯一标识符（UUID）
        content: 父块完整内容
        filename: 所属文件名
        heading_path: 标题路径（如 ["第一章", "1.1 节"]）
        page_start: 起始页码
        page_end: 结束页码
        created_at: 创建时间
    """
    __tablename__ = "parent_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False, index=True)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SessionORM(Base):
    """
    会话 ORM 模型

    存储对话会话的元信息。

    属性:
        id: 会话 ID（16 位十六进制字符串）
        title: 会话标题
        summary: 滚动对话摘要（用于长对话压缩）
        created_at: 创建时间
        updated_at: 最后更新时间
    """
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新对话")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MessageORM(Base):
    """
    消息 ORM 模型

    存储对话中的每条消息。

    属性:
        id: 消息 ID（自增主键）
        session_id: 所属会话 ID
        role: 消息角色（user/assistant）
        content: 消息内容
        sources: 关联的来源信息（JSONB）
        created_at: 创建时间
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserORM(Base):
    """
    用户 ORM 模型

    存储用户账号信息，用于 JWT 认证。

    属性:
        id: 用户 ID（自增主键）
        username: 用户名（唯一）
        hashed_password: bcrypt 哈希后的密码
        created_at: 账号创建时间
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvalRunORM(Base):
    """
    评估运行记录 ORM 模型

    存储每次评估运行的汇总信息。

    属性:
        id: 运行 ID（UUID）
        strategy: 检索策略（fast/precise/deep）
        dataset_name: 数据集名称
        question_count: 评估问题数量
        status: 运行状态（running/completed/failed）
        started_at: 开始时间
        completed_at: 完成时间
        error_message: 错误信息
        avg_faithfulness: 平均忠实度
        avg_context_recall: 平均上下文召回率
        avg_context_precision: 平均上下文精确度
        avg_answer_relevancy: 平均答案相关性
    """
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(200), nullable=False)
    question_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    avg_faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)


class EvalResultORM(Base):
    """
    评估结果详情 ORM 模型

    存储每条 QA 对的评估结果。

    属性:
        id: 自增主键
        run_id: 所属运行 ID
        question: 问题文本
        ground_truth: 标准答案
        answer: 生成的答案
        strategy: 实际使用的检索策略
        contexts: 检索到的上下文列表（JSONB）
        faithfulness: 忠实度
        context_recall: 上下文召回率
        context_precision: 上下文精确度
        answer_relevancy: 答案相关性
    """
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str | None] = mapped_column(String(20), nullable=True)
    contexts: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
