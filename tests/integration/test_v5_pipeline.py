import os
import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.db_models import Base
from backend.services.parent_store import ParentStore
from backend.services.parsing.markdown_parser import MarkdownParser
from backend.services.chunking.hierarchical_chunker import HierarchicalChunker

SAMPLE_MD = """# Test Document

Introduction paragraph here.

## Section One

Some detailed content for section one.

## Section Two

More content in section two.

### Subsection 2.1

Nested detail here.
"""


@pytest.fixture(scope="module")
def pg_container():
    with PostgresContainer("postgres:17") as pc:
        yield pc


@pytest.fixture
def store(pg_container):
    engine = create_engine(pg_container.get_connection_url())
    Base.metadata.create_all(engine)
    yield ParentStore(sessionmaker(engine))


def test_parse_chunk_store_retrieve(store):
    """End-to-end: parse markdown -> chunk -> store parents in PG -> retrieve by parent_id."""
    parser = MarkdownParser()
    elements = parser.parse_string(SAMPLE_MD)

    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "test.md")

    assert len(parents) >= 2  # H2 sections (plus possible lead-in)
    assert len(leaves) >= 2

    # Store parents
    store.add(parents)

    # Retrieve one parent by leaf's parent_id
    leaf_parent_id = leaves[0].parent_id
    fetched = store.get_by_ids([leaf_parent_id])
    assert len(fetched) == 1
    assert fetched[0].id == leaf_parent_id

    # Verify heading_path
    assert len(fetched[0].heading_path) >= 1

    # Verify parent content is longer than individual leaf (parent-child check)
    leaf_content = leaves[0].content
    parent_content = fetched[0].content
    assert len(parent_content) >= len(leaf_content)
