import sqlite3
import uuid
import json
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

DB_PATH = "data/sessions.db"


class SessionService:
    def __init__(self):
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '新对话',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.commit()

    def create_session(self, title: str = "新对话") -> str:
        session_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    def list_sessions(self) -> list[dict]:
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
        with self._get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None):
        now = datetime.now().isoformat()
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
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
        """Return messages with sources for API response."""
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
        with self._get_conn() as conn:
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            conn.commit()


session_service = SessionService()
