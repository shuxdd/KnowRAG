"""
会话服务模块

管理对话会话（Session）和消息（Message）的持久化存储。
使用 PostgreSQL 数据库存储，支持多轮对话。

主要功能：
- 创建会话：create_session()
- 获取会话：get_session(), list_sessions()
- 删除会话：delete_session()
- 消息管理：add_message(), get_history(), get_messages()
- 会话标题：update_title()
- 记忆压缩：get_summary(), update_summary()

数据结构：
- sessions 表：存储会话元信息（ID、标题、摘要、创建时间、更新时间）
- messages 表：存储对话消息（角色、内容、来源、创建时间）
"""

import uuid
import logging
from datetime import datetime, timezone

from sqlalchemy import func, desc
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from backend.db import SessionFactory
from backend.models.db_models import SessionORM, MessageORM

logger = logging.getLogger(__name__)


class SessionService:
    """会话管理服务，负责会话和消息的持久化存储，使用 PostgreSQL 数据库。"""

    def create_session(self, title: str = "新对话", user_id: int = 0) -> str:
        session_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc)
        with SessionFactory() as db:
            db.add(SessionORM(
                id=session_id,
                title=title,
                user_id=user_id,
                created_at=now,
                updated_at=now,
            ))
            db.commit()
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        with SessionFactory() as db:
            row = db.query(SessionORM).filter(SessionORM.id == session_id).first()
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def list_sessions(self, user_id: int | None = None) -> list[dict]:
        with SessionFactory() as db:
            query = db.query(
                SessionORM.id,
                SessionORM.title,
                SessionORM.created_at,
                SessionORM.updated_at,
                func.count(MessageORM.id).label("msg_count"),
            ).outerjoin(MessageORM, SessionORM.id == MessageORM.session_id)
            if user_id is not None:
                query = query.filter(SessionORM.user_id == user_id)
            rows = query.group_by(SessionORM.id).order_by(desc(SessionORM.updated_at)).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "message_count": r.msg_count,
            }
            for r in rows
        ]

    def delete_session(self, session_id: str, user_id: int | None = None) -> bool:
        with SessionFactory() as db:
            query = db.query(SessionORM).filter(SessionORM.id == session_id)
            if user_id is not None:
                query = query.filter(SessionORM.user_id == user_id)
            session = query.first()
            if not session:
                return False
            db.query(MessageORM).filter(MessageORM.session_id == session_id).delete()
            db.delete(session)
            db.commit()
            return True

    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None):
        now = datetime.now(timezone.utc)
        with SessionFactory() as db:
            db.add(MessageORM(
                session_id=session_id,
                role=role,
                content=content,
                sources=sources,
                created_at=now,
            ))
            db.query(SessionORM).filter(SessionORM.id == session_id).update(
                {"updated_at": now}
            )
            db.commit()

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        history = InMemoryChatMessageHistory()
        with SessionFactory() as db:
            rows = (
                db.query(MessageORM.role, MessageORM.content)
                .filter(MessageORM.session_id == session_id)
                .order_by(MessageORM.id.asc())
                .all()
            )
        for role, content in rows:
            if role == "user":
                history.add_message(HumanMessage(content=content))
            else:
                history.add_message(AIMessage(content=content))
        return history

    def get_messages(self, session_id: str) -> list[dict]:
        with SessionFactory() as db:
            rows = (
                db.query(MessageORM)
                .filter(MessageORM.session_id == session_id)
                .order_by(MessageORM.id.asc())
                .all()
            )
        return [
            {
                "role": r.role,
                "content": r.content,
                "sources": r.sources,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    def update_title(self, session_id: str, title: str):
        with SessionFactory() as db:
            db.query(SessionORM).filter(SessionORM.id == session_id).update(
                {"title": title}
            )
            db.commit()

    def get_summary(self, session_id: str) -> str:
        with SessionFactory() as db:
            row = db.query(SessionORM.summary).filter(SessionORM.id == session_id).first()
        return (row[0] or "") if row else ""

    def update_summary(self, session_id: str, summary: str):
        with SessionFactory() as db:
            db.query(SessionORM).filter(SessionORM.id == session_id).update(
                {"summary": summary}
            )
            db.commit()


session_service = SessionService()
