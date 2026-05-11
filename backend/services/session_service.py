import sqlite3
import uuid
import json
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

DB_PATH = "data/sessions.db"


class SessionService:
    """
    会话管理服务
    负责会话（Session）和消息（Message）的持久化存储
    使用 SQLite 数据库存储，支持多轮对话
    """

    def __init__(self):
        """
        初始化会话服务
        创建数据库连接和表结构
        """
        self._init_db()

    def _get_conn(self):
        """
        获取数据库连接

        Returns:
            sqlite3.Connection 对象
        """
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        """
        初始化数据库表结构
        创建 sessions 表（存储会话元信息）和 messages 表（存储对话消息）
        """
        with self._get_conn() as conn:
            # 会话表：存储会话的基本信息
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,        -- 会话 ID
                    title TEXT DEFAULT '新对话', -- 会话标题
                    created_at TEXT NOT NULL,    -- 创建时间
                    updated_at TEXT NOT NULL     -- 最后更新时间
                )
            """)
            # 消息表：存储会话中的每条消息
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 消息 ID
                    session_id TEXT NOT NULL,              -- 所属会话 ID
                    role TEXT NOT NULL,                   -- 角色（user/assistant）
                    content TEXT NOT NULL,                -- 消息内容
                    sources TEXT,                          -- 消息关联的来源信息（JSON）
                    created_at TEXT NOT NULL,              -- 创建时间
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.commit()

    def create_session(self, title: str = "新对话") -> str:
        """
        创建新会话

        Args:
            title: 会话标题，默认为"新对话"

        Returns:
            新创建的会话 ID
        """
        session_id = uuid.uuid4().hex[:16]  # 生成16位十六进制会话 ID
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        """
        获取指定会话的信息

        Args:
            session_id: 会话 ID

        Returns:
            会话信息字典，如果不存在则返回 None
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    def list_sessions(self) -> list[dict]:
        """
        获取所有会话列表

        Returns:
            会话列表，按最后更新时间降序排列
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) as msg_count
                   FROM sessions s LEFT JOIN messages m ON s.id = m.session_id
                   GROUP BY s.id ORDER BY s.updated_at DESC"""
            ).fetchall()
        return [
            {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3], "message_count": r[4]}
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话及其所有消息

        Args:
            session_id: 会话 ID

        Returns:
            是否成功删除（会话存在返回 True）
        """
        with self._get_conn() as conn:
            # 先删除该会话的所有消息
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # 再删除会话本身
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None):
        """
        向指定会话添加消息

        Args:
            session_id: 会话 ID
            role: 消息角色（user/assistant）
            content: 消息内容
            sources: 关联的来源信息列表
        """
        now = datetime.now().isoformat()
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, now),
            )
            # 更新会话的最后更新时间
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        获取指定会话的对话历史（用于 LLM 上下文）

        Args:
            session_id: 会话 ID

        Returns:
            LangChain 格式的聊天历史对象
        """
        history = InMemoryChatMessageHistory()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        for role, content in rows:
            if role == "user":
                history.add_message(HumanMessage(content=content))
            else:
                history.add_message(AIMessage(content=content))
        return history

    def get_messages(self, session_id: str) -> list[dict]:
        """
        获取指定会话的所有消息（用于 API 返回）

        Args:
            session_id: 会话 ID

        Returns:
            消息列表，每条消息包含 role、content、sources 和 created_at
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        result = []
        for role, content, sources_json, created_at in rows:
            sources = json.loads(sources_json) if sources_json else None
            result.append({
                "role": role,
                "content": content,
                "sources": sources,
                "created_at": created_at,
            })
        return result

    def update_title(self, session_id: str, title: str):
        """
        更新指定会话的标题

        Args:
            session_id: 会话 ID
            title: 新标题
        """
        with self._get_conn() as conn:
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            conn.commit()


# 全局单例实例
session_service = SessionService()
