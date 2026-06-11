"""Tests for graph_service — uses mocked Neo4j driver."""

import sys
import pytest
from unittest.mock import MagicMock, patch


# Patch neo4j.GraphDatabase.driver so the module-level singleton
# does not attempt to connect to a real Neo4j instance on import.
_mock_neo4j_driver = MagicMock()

with patch("neo4j.GraphDatabase.driver", return_value=_mock_neo4j_driver):
    from backend.services.graph_service import GraphService


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver with session support."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


def test_build_graph_for_chunk_merges_entities(mock_driver):
    """build_graph_for_chunk should MERGE entities and relationships."""
    driver, session = mock_driver

    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    entities = [
        {"name": "BGE", "type": "技术", "description": "文本嵌入模型"},
        {"name": "Milvus", "type": "技术", "description": "向量数据库"},
    ]
    relations = [
        {"source": "BGE", "target": "Milvus", "relation": "依赖", "context": "BGE 用于 Milvus 的向量检索"},
    ]

    svc.build_graph_for_chunk(
        chunk_id="chunk-1",
        filename="test.md",
        heading_path='["第一章"]',
        entities=entities,
        relations=relations,
        user_id=1,
    )

    # Should have called session.run at least 3 times:
    # 1 for ParentChunk MERGE, 2 for entity MERGEs, 1 for relation MERGE
    assert session.run.call_count >= 3


def test_search_by_entities_returns_chunk_ids(mock_driver):
    """search_by_entities should return parent chunk ids from graph traversal."""
    driver, session = mock_driver

    # Mock the result of the Cypher query
    record1 = MagicMock()
    record1.__getitem__ = MagicMock(side_effect=lambda k: "chunk-1" if k == "chunk_id" else 1.0)
    record2 = MagicMock()
    record2.__getitem__ = MagicMock(side_effect=lambda k: "chunk-2" if k == "chunk_id" else 0.5)

    result_mock = MagicMock()
    result_mock.__iter__ = MagicMock(return_value=iter([record1, record2]))
    session.run.return_value = result_mock

    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    chunk_ids = svc.search_by_entities(["BGE", "Milvus"], user_id=1, top_k=5)
    assert chunk_ids == ["chunk-1", "chunk-2"]


def test_delete_by_filename(mock_driver):
    """delete_by_filename should clean up chunks, relations, and orphan entities."""
    driver, session = mock_driver

    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    svc.delete_by_filename("test.md", user_id=1, chunk_ids=["chunk-1", "chunk-2"])

    # Should call session.run for cleanup
    assert session.run.call_count >= 2


def test_get_stats(mock_driver):
    """get_stats should return entity/relation/type counts."""
    driver, session = mock_driver

    record = MagicMock()
    record.__getitem__ = MagicMock(side_effect=lambda k: {
        "entity_count": 10, "relation_count": 15, "type_count": 3
    }[k])
    result_mock = MagicMock()
    result_mock.single.return_value = record
    session.run.return_value = result_mock

    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    stats = svc.get_stats(user_id=1)
    assert stats["entity_count"] == 10
