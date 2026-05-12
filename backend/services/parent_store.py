import uuid
from backend.db import SessionFactory
from backend.models.db_models import ParentChunkORM
from backend.models.chunk_types import ParentChunk


class ParentStore:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def add(self, parents: list[ParentChunk]) -> None:
        with self._session_factory() as session:
            for p in parents:
                orm = ParentChunkORM(
                    id=uuid.UUID(p.id),
                    content=p.content,
                    filename=p.filename,
                    heading_path=p.heading_path,
                    page_start=p.page_start,
                    page_end=p.page_end,
                )
                session.add(orm)
            session.commit()

    def get_by_ids(self, ids: list[str]) -> list[ParentChunk]:
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
                )
                for i in valid_ids if (r := id_map.get(i))
            ]

    def delete_by_ids(self, ids: list[str]) -> int:
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

    def delete_by_filename(self, filename: str) -> int:
        with self._session_factory() as session:
            count = session.query(ParentChunkORM).filter(ParentChunkORM.filename == filename).delete()
            session.commit()
            return count


parent_store = ParentStore(SessionFactory)
