import uuid
import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.db_models import Base
from backend.services.parent_store import ParentStore


@pytest.fixture(scope="module")
def pg_container():
    with PostgresContainer("postgres:17-alpine") as pc:
        yield pc


@pytest.fixture
def store(pg_container):
    engine = create_engine(pg_container.get_connection_url())
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    maker = sessionmaker(engine)
    return ParentStore(maker)


@pytest.fixture
def sample_parents():
    from backend.models.chunk_types import ParentChunk

    parent = ParentChunk(
        id=str(uuid.uuid4()),
        content="Sample parent content",
        heading_path=["Chapter 1", "Section 1.1"],
        filename="test.md",
        page_start=1,
        page_end=2,
    )
    return [parent]


class TestParentStore:
    def test_add_and_get_by_ids(self, store, sample_parents):
        store.add(sample_parents)
        fetched = store.get_by_ids([sample_parents[0].id])
        assert len(fetched) == 1
        assert fetched[0].content == "Sample parent content"

    def test_get_by_ids_empty(self, store):
        assert store.get_by_ids(["nonexistent"]) == []

    def test_delete_by_ids(self, store, sample_parents):
        store.add(sample_parents)
        count = store.delete_by_ids([sample_parents[0].id])
        assert count == 1
        assert store.get_by_ids([sample_parents[0].id]) == []

    def test_delete_by_filename(self, store, sample_parents):
        store.add(sample_parents)
        count = store.delete_by_filename("test.md")
        assert count == 1
        assert store.get_by_ids([sample_parents[0].id]) == []
