"""
父块存储服务模块

负责在 PostgreSQL 数据库中存储和管理父块（ParentChunk）数据。

主要功能：
- add(): 添加父块到数据库
- get_by_ids(): 根据 ID 列表批量获取父块
- get_by_filename(): 根据文件名获取该文件的所有父块
- delete_by_ids(): 根据 ID 列表删除父块
- delete_by_filename(): 根据文件名删除父块
"""

import uuid
from backend.db import SessionFactory
from backend.models.db_models import ParentChunkORM
from backend.models.chunk_types import ParentChunk


class ParentStore:
    """
    父块存储服务

    提供父块的 CRUD 操作，与 PostgreSQL 数据库交互。
    """

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def add(self, parents: list[ParentChunk], user_id: int = 0) -> None:
        """
        添加父块列表到数据库

        Args:
            parents: 父块对象列表
            user_id: 所属用户 ID
        """
        with self._session_factory() as session:
            for p in parents:
                orm = ParentChunkORM(
                    id=uuid.UUID(p.id),
                    content=p.content,
                    filename=p.filename,
                    heading_path=p.heading_path,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    user_id=user_id,
                    created_at=p.created_at,
                )
                session.add(orm)
            session.commit()

    def get_by_ids(self, ids: list[str]) -> list[ParentChunk]:
        """
        根据 ID 列表批量获取父块

        Args:
            ids: 父块 ID 列表

        Returns:
            父块对象列表（按输入顺序）
        """
        uuids = []
        valid_ids = []
        for i in ids:
            try:
                uuids.append(uuid.UUID(i))
                valid_ids.append(i)
            except ValueError:
                continue
        if not uuids:
            return []
        with self._session_factory() as session:
            rows = session.query(ParentChunkORM).filter(ParentChunkORM.id.in_(uuids)).all()
            id_map = {str(r.id): r for r in rows}
            return [
                ParentChunk(
                    id=str(r.id),
                    content=r.content,
                    filename=r.filename,
                    heading_path=r.heading_path,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    created_at=r.created_at,
                )
                for i in valid_ids if (r := id_map.get(i))
            ]

    def delete_by_ids(self, ids: list[str]) -> int:
        """
        根据 ID 列表删除父块

        Args:
            ids: 父块 ID 列表

        Returns:
            删除的父块数量
        """
        uuids = []
        for i in ids:
            try:
                uuids.append(uuid.UUID(i))
            except ValueError:
                continue
        if not uuids:
            return 0
        with self._session_factory() as session:
            count = session.query(ParentChunkORM).filter(ParentChunkORM.id.in_(uuids)).delete()
            session.commit()
            return count

    def get_by_filename(self, filename: str, user_id: int | None = None) -> list[ParentChunk]:
        """
        根据文件名获取该文件的所有父块

        Args:
            filename: 文件名
            user_id: 所属用户 ID（可选，为 None 时不按用户过滤）

        Returns:
            父块对象列表
        """
        with self._session_factory() as session:
            query = session.query(ParentChunkORM).filter(ParentChunkORM.filename == filename)
            if user_id is not None:
                query = query.filter(ParentChunkORM.user_id == user_id)
            rows = query.all()
            return [
                ParentChunk(
                    id=str(r.id),
                    content=r.content,
                    filename=r.filename,
                    heading_path=r.heading_path,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    created_at=r.created_at,
                )
                for r in rows
            ]

    def delete_by_filename(self, filename: str, user_id: int | None = None) -> int:
        """
        根据文件名删除父块

        Args:
            filename: 文件名
            user_id: 所属用户 ID（可选）

        Returns:
            删除的父块数量
        """
        with self._session_factory() as session:
            query = session.query(ParentChunkORM).filter(ParentChunkORM.filename == filename)
            if user_id is not None:
                query = query.filter(ParentChunkORM.user_id == user_id)
            count = query.delete()
            session.commit()
            return count


parent_store = ParentStore(SessionFactory)
