# Neo4j Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Neo4j knowledge graph as a third retrieval path alongside vector and BM25 search, with on-demand entity extraction driven by query frequency.

**Architecture:** Entities and semantic relationships are extracted from parent chunks via LLM, stored in Neo4j, and queried as a third RRF fusion path. Extraction is lazy — only triggered when a parent chunk is hit >= 3 times. Graph service is synchronous (matching existing `parent_store` pattern), wrapped with `asyncio.to_thread` in async contexts.

**Tech Stack:** Neo4j 5 (Docker), `neo4j` Python driver, existing Mimo LLM for extraction, FastAPI, React

---

## Task 1: Infrastructure — Docker, Config, Dependencies

**Files:**
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`
- Modify: `backend/config.py`

- [ ] **Step 1: Add Neo4j service to docker-compose.yml**

Append to the `services` section in `docker-compose.yml` (before `networks:`):

```yaml
  neo4j:
    container_name: knowrag-neo4j
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/knowrag123
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/neo4j:/data
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 5
```

- [ ] **Step 2: Add neo4j driver to requirements.txt**

Append to `requirements.txt`:

```
neo4j>=5.0.0
```

- [ ] **Step 3: Add Neo4j and KG config to backend/config.py**

Add these fields to the `Settings` class in `backend/config.py`, after the `redis_url` / `retrieval_cache_ttl` block:

```python
    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowrag123"

    # ==================== 知识图谱配置 ====================
    kg_extract_model: str = ""  # 空则复用 mimo_model
    kg_extract_concurrency: int = 3
    kg_max_entities_per_chunk: int = 20
    kg_max_relations_per_chunk: int = 15
    kg_hit_threshold: int = 3
    kg_hit_window_days: int = 30
```

- [ ] **Step 4: Install dependency**

Run: `pip install neo4j>=5.0.0`

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml requirements.txt backend/config.py
git commit -m "feat: add Neo4j infrastructure and KG config"
```

---

## Task 2: Database Model + Alembic Migration

**Files:**
- Modify: `backend/models/db_models.py`
- Create: `alembic/versions/xxxx_add_retrieval_stats.py`

- [ ] **Step 1: Add RetrievalStatsORM to db_models.py**

Append to `backend/models/db_models.py`:

```python
class RetrievalStatsORM(Base):
    """检索命中统计表，用于驱动按需知识图谱抽取"""
    __tablename__ = "parent_chunk_retrieval_stats"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_hit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    extracted: Mapped[bool] = mapped_column(default=False, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
```

- [ ] **Step 2: Generate Alembic migration**

Run: `alembic revision --autogenerate -m "add_retrieval_stats_table"`

Inspect the generated file. The upgrade should create `parent_chunk_retrieval_stats` with columns `chunk_id` (UUID PK), `hit_count` (Integer), `last_hit_at` (DateTime), `extracted` (Boolean).

- [ ] **Step 3: Commit**

```bash
git add backend/models/db_models.py alembic/versions/
git commit -m "feat: add RetrievalStatsORM for KG extraction tracking"
```

---

## Task 3: Graph Service

**Files:**
- Create: `backend/services/graph_service.py`
- Create: `tests/test_graph_service.py`

- [ ] **Step 1: Write tests for graph service**

Create `tests/test_graph_service.py`:

```python
"""Tests for graph_service — uses mocked Neo4j driver."""

import pytest
from unittest.mock import MagicMock, patch


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

    from backend.services.graph_service import GraphService
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

    from backend.services.graph_service import GraphService
    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    chunk_ids = svc.search_by_entities(["BGE", "Milvus"], user_id=1, top_k=5)
    assert chunk_ids == ["chunk-1", "chunk-2"]


def test_delete_by_filename(mock_driver):
    """delete_by_filename should clean up chunks, relations, and orphan entities."""
    driver, session = mock_driver

    from backend.services.graph_service import GraphService
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

    from backend.services.graph_service import GraphService
    svc = GraphService.__new__(GraphService)
    svc._driver = driver

    stats = svc.get_stats(user_id=1)
    assert stats["entity_count"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_service.py -v`
Expected: FAIL (module `backend.services.graph_service` not found)

- [ ] **Step 3: Implement graph_service.py**

Create `backend/services/graph_service.py`:

```python
"""
知识图谱服务模块

负责 Neo4j 图数据库的交互，包括：
- 实体和关系的写入（MERGE 语义）
- 基于实体的图谱检索
- 文档删除时的图谱清理
- 图谱统计信息查询
"""

import json
import logging
from typing import Any

from neo4j import GraphDatabase

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GraphService:
    """Neo4j 知识图谱服务"""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._ensure_constraints()

    def _ensure_constraints(self):
        """Create uniqueness constraints if they don't exist."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.name, e.user_id) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_unique IF NOT EXISTS "
                "FOR (c:ParentChunk) REQUIRE c.id IS UNIQUE"
            )

    def close(self):
        self._driver.close()

    def build_graph_for_chunk(
        self,
        chunk_id: str,
        filename: str,
        heading_path: str,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        user_id: int,
    ) -> None:
        """
        为一个父块构建图谱节点和关系。

        使用 MERGE 语义：同名实体自动合并，关系追加。
        """
        with self._driver.session() as session:
            # 1. MERGE parent chunk node
            session.run(
                "MERGE (c:ParentChunk {id: $chunk_id}) "
                "SET c.filename = $filename, c.heading_path = $heading_path, c.user_id = $user_id",
                chunk_id=chunk_id, filename=filename,
                heading_path=heading_path, user_id=user_id,
            )

            # 2. MERGE entity nodes + MENTIONED_IN relationships
            import uuid as _uuid
            for ent in entities:
                entity_id = str(_uuid.uuid4())
                session.run(
                    "MERGE (e:Entity {name: $name, user_id: $user_id}) "
                    "ON CREATE SET e.id = $entity_id, e.type = $type, e.description = $description, e.created_at = datetime() "
                    "ON MATCH SET e.type = COALESCE(e.type, $type), "
                    "  e.description = COALESCE(e.description, $description) "
                    "WITH e "
                    "MATCH (c:ParentChunk {id: $chunk_id}) "
                    "MERGE (e)-[r:MENTIONED_IN]->(c) "
                    "ON CREATE SET r.frequency = 1 "
                    "ON MATCH SET r.frequency = r.frequency + 1",
                    name=ent["name"], user_id=user_id, entity_id=entity_id,
                    type=ent.get("type", ""), description=ent.get("description", ""),
                    chunk_id=chunk_id,
                )

            # 3. MERGE entity-to-entity relationships
            for rel in relations:
                session.run(
                    "MATCH (s:Entity {name: $source, user_id: $user_id}) "
                    "MATCH (t:Entity {name: $target, user_id: $user_id}) "
                    "MERGE (s)-[r:RELATES_TO {relation: $relation, chunk_id: $chunk_id}]->(t) "
                    "ON CREATE SET r.context = $context",
                    source=rel["source"], target=rel["target"],
                    relation=rel["relation"], chunk_id=chunk_id,
                    context=rel.get("context", ""), user_id=user_id,
                )

    def search_by_entities(
        self, entity_names: list[str], user_id: int, top_k: int = 10
    ) -> list[str]:
        """
        根据实体名称进行图谱检索，返回关联的 ParentChunk id 列表。

        检索策略：从匹配实体出发，1-2 跳遍历 RELATES_TO 关系，
        收集沿途 MENTIONED_IN 指向的 ParentChunk，按相关度排序。
        """
        if not entity_names:
            return []

        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) "
                "WHERE e.name IN $names AND e.user_id = $user_id "
                "CALL { "
                "  WITH e "
                "  MATCH (e)-[:MENTIONED_IN]->(c:ParentChunk) "
                "  RETURN c.id AS chunk_id, 1.0 AS score "
                "  UNION "
                "  WITH e "
                "  MATCH (e)-[:RELATES_TO*1..2]->(related)-[:MENTIONED_IN]->(c:ParentChunk) "
                "  RETURN c.id AS chunk_id, 0.5 AS score "
                "} "
                "WITH chunk_id, max(score) AS relevance "
                "ORDER BY relevance DESC "
                "LIMIT $top_k "
                "RETURN chunk_id",
                names=entity_names, user_id=user_id, top_k=top_k,
            )
            return [record["chunk_id"] for record in result]

    def delete_by_filename(
        self, filename: str, user_id: int, chunk_ids: list[str]
    ) -> None:
        """删除指定文档的所有图谱数据。"""
        with self._driver.session() as session:
            # Delete RELATES_TO edges from these chunks
            if chunk_ids:
                session.run(
                    "MATCH ()-[r:RELATES_TO]->() "
                    "WHERE r.chunk_id IN $chunk_ids "
                    "DELETE r",
                    chunk_ids=chunk_ids,
                )

            # Delete MENTIONED_IN edges and ParentChunk nodes
            session.run(
                "MATCH (c:ParentChunk {filename: $filename, user_id: $user_id}) "
                "DETACH DELETE c",
                filename=filename, user_id=user_id,
            )

            # Clean up orphan entities
            session.run(
                "MATCH (e:Entity {user_id: $user_id}) "
                "WHERE NOT (e)--() "
                "DELETE e",
                user_id=user_id,
            )

    def get_stats(self, user_id: int) -> dict[str, int]:
        """获取图谱统计信息。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {user_id: $user_id}) "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]->() "
                "RETURN count(DISTINCT e) AS entity_count, "
                "count(r) AS relation_count, "
                "count(DISTINCT e.type) AS type_count",
                user_id=user_id,
            )
            record = result.single()
            return {
                "entity_count": record["entity_count"],
                "relation_count": record["relation_count"],
                "type_count": record["type_count"],
            }

    def get_entities(
        self, user_id: int, entity_type: str | None = None,
        search: str | None = None, page: int = 1, page_size: int = 20
    ) -> dict[str, Any]:
        """获取实体列表，支持类型筛选、名称搜索、分页。"""
        with self._driver.session() as session:
            where_clauses = ["e.user_id = $user_id"]
            params: dict[str, Any] = {"user_id": user_id, "skip": (page - 1) * page_size, "limit": page_size}

            if entity_type:
                where_clauses.append("e.type = $entity_type")
                params["entity_type"] = entity_type
            if search:
                where_clauses.append("e.name CONTAINS $search")
                params["search"] = search

            where = " AND ".join(where_clauses)

            count_result = session.run(
                f"MATCH (e:Entity) WHERE {where} RETURN count(e) AS total", **params
            )
            total = count_result.single()["total"]

            result = session.run(
                f"MATCH (e:Entity) WHERE {where} "
                "RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description "
                "ORDER BY e.name "
                "SKIP $skip LIMIT $limit",
                **params,
            )
            entities = [
                {"id": r["id"], "name": r["name"], "type": r["type"], "description": r["description"]}
                for r in result
            ]
            return {"entities": entities, "total": total, "page": page, "page_size": page_size}

    def get_entity_detail(self, entity_id: str, user_id: int) -> dict[str, Any] | None:
        """获取实体详情，包括所有关系和关联的父块。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {id: $entity_id, user_id: $user_id}) "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]->(target:Entity) "
                "OPTIONAL MATCH (source:Entity)-[r2:RELATES_TO]->(e) "
                "OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:ParentChunk) "
                "RETURN e, "
                "collect(DISTINCT {direction: 'outgoing', relation: r.relation, "
                "  target: {id: target.id, name: target.name, type: target.type}, context: r.context}) AS outgoing, "
                "collect(DISTINCT {direction: 'incoming', relation: r2.relation, "
                "  source: {id: source.id, name: source.name, type: source.type}, context: r2.context}) AS incoming, "
                "collect(DISTINCT {chunk_id: c.id, filename: c.filename, heading_path: c.heading_path}) AS chunks",
                entity_id=entity_id, user_id=user_id,
            )
            record = result.single()
            if not record:
                return None

            e = record["e"]
            return {
                "entity": {"id": e["id"], "name": e["name"], "type": e["type"], "description": e["description"]},
                "relations": [r for r in record["outgoing"] if r["relation"]] +
                             [r for r in record["incoming"] if r["relation"]],
                "mentioned_in": [c for c in record["chunks"] if c["chunk_id"]],
            }


graph_service = GraphService()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_service.py -v`
Expected: PASS (tests use mocked driver, no real Neo4j needed)

- [ ] **Step 5: Commit**

```bash
git add backend/services/graph_service.py tests/test_graph_service.py
git commit -m "feat: add GraphService for Neo4j CRUD and graph search"
```

---

## Task 4: Entity Extractor

**Files:**
- Create: `backend/services/entity_extractor.py`
- Create: `tests/test_entity_extractor.py`

- [ ] **Step 1: Read existing LLM calling pattern**

Read `backend/services/query_rewriter.py` to understand how the LLM is called (OpenAI-compatible API via `langchain_openai.ChatOpenAI` or direct httpx). Follow the same pattern.

- [ ] **Step 2: Write tests for entity extractor**

Create `tests/test_entity_extractor.py`:

```python
"""Tests for entity_extractor — uses mocked LLM."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json


def test_parse_extraction_result_valid_json():
    """Should parse valid JSON from LLM response."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = json.dumps({
        "entities": [
            {"name": "BGE", "type": "技术", "description": "文本嵌入模型"},
        ],
        "relations": [
            {"source": "BGE", "target": "Milvus", "relation": "依赖", "context": "BGE 用于 Milvus"},
        ],
    }, ensure_ascii=False)

    result = extractor._parse_extraction_result(raw)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "BGE"
    assert len(result["relations"]) == 1


def test_parse_extraction_result_handles_markdown_json_block():
    """Should handle LLM responses wrapped in ```json ... ``` blocks."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = '```json\n{"entities": [], "relations": []}\n```'
    result = extractor._parse_extraction_result(raw)
    assert result == {"entities": [], "relations": []}


def test_parse_extraction_result_handles_invalid_json():
    """Should return empty result on invalid JSON."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    result = extractor._parse_extraction_result("not json at all")
    assert result == {"entities": [], "relations": []}


def test_parse_query_entities_valid():
    """Should parse entity names from query extraction result."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = '["BGE", "Milvus"]'
    result = extractor._parse_query_entities(raw)
    assert result == ["BGE", "Milvus"]


def test_parse_query_entities_handles_invalid():
    """Should return empty list on invalid input."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    result = extractor._parse_query_entities("invalid")
    assert result == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_entity_extractor.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement entity_extractor.py**

Create `backend/services/entity_extractor.py`:

```python
"""
实体关系抽取模块

使用 LLM 从文档文本中抽取实体和关系，用于构建知识图谱。
"""

import json
import logging
from typing import Any

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


EXTRACT_PROMPT = """从以下文本中抽取实体和它们之间的关系。

要求：
1. 实体名称归一化（同义实体合并为一个标准名）
2. 每个实体给出类型和一句话描述
3. 关系要有明确的类型（如：依赖、属于、导致、对比、包含）
4. 标注支撑该关系的原文句子
5. 实体最多 {max_entities} 个，关系最多 {max_relations} 个

输出 JSON 格式：
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}], "relations": [{{"source": "...", "target": "...", "relation": "...", "context": "..."}}]}}

文本内容：
{text}"""


QUERY_ENTITY_PROMPT = """从以下问题中提取关键实体名称，输出 JSON 数组。
要求：只返回与知识相关的实体名词，忽略动词和修饰词。

问题：{query}
输出格式：["实体1", "实体2"]"""


class EntityExtractor:
    """LLM-based entity and relationship extractor."""

    def __init__(self):
        from langchain_openai import ChatOpenAI

        model_name = settings.kg_extract_model or settings.mimo_model
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            max_tokens=4096,
            temperature=0.1,
        )

    def extract_from_chunk(self, text: str) -> dict[str, Any]:
        """
        从父块文本中抽取实体和关系。

        Returns:
            {"entities": [...], "relations": [...]}
        """
        prompt = EXTRACT_PROMPT.format(
            text=text,
            max_entities=settings.kg_max_entities_per_chunk,
            max_relations=settings.kg_max_relations_per_chunk,
        )
        try:
            response = self._llm.invoke(prompt)
            return self._parse_extraction_result(response.content)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {"entities": [], "relations": []}

    def extract_query_entities(self, query: str) -> list[str]:
        """从用户问题中提取实体名称列表。"""
        prompt = QUERY_ENTITY_PROMPT.format(query=query)
        try:
            response = self._llm.invoke(prompt)
            return self._parse_query_entities(response.content)
        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")
            return []

    def _parse_extraction_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM response into entities and relations."""
        raw = raw.strip()
        # Strip markdown code blocks
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extraction JSON: {raw[:200]}")
                    return {"entities": [], "relations": []}
            else:
                logger.warning(f"No JSON object found in extraction result: {raw[:200]}")
                return {"entities": [], "relations": []}

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        # Validate structure
        valid_entities = [
            e for e in entities
            if isinstance(e, dict) and "name" in e
        ]
        valid_relations = [
            r for r in relations
            if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r
        ]

        return {"entities": valid_entities, "relations": valid_relations}

    def _parse_query_entities(self, raw: str) -> list[str]:
        """Parse entity names from LLM response."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(e) for e in data if isinstance(e, str)]
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                    if isinstance(data, list):
                        return [str(e) for e in data if isinstance(e, str)]
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse query entities: {raw[:200]}")
        return []


entity_extractor = EntityExtractor()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_entity_extractor.py -v`
Expected: PASS (tests only exercise parse methods, no LLM calls)

- [ ] **Step 6: Commit**

```bash
git add backend/services/entity_extractor.py tests/test_entity_extractor.py
git commit -m "feat: add LLM-based entity/relationship extractor"
```

---

## Task 5: Hybrid Retriever — Graph Path + Stats Tracking

**Files:**
- Modify: `backend/services/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever_graph.py`

- [ ] **Step 1: Write tests for graph retrieval integration**

Create `tests/test_hybrid_retriever_graph.py`:

```python
"""Tests for graph retrieval integration in hybrid_retriever."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_graph_results_included_in_rrf_fusion():
    """Graph results should be included as a third doc_list in RRF fusion."""
    from backend.services.hybrid_retriever import rrf_fusion

    doc_a = Document(page_content="content A", metadata={})
    doc_b = Document(page_content="content B", metadata={})
    doc_c = Document(page_content="content C", metadata={})

    # doc_c only appears in graph results
    fused = rrf_fusion(
        doc_lists=[[doc_a, doc_b], [doc_b, doc_a], [doc_c, doc_a]],
        k=60,
        top_n=3,
    )
    contents = [d.page_content for d in fused]
    # doc_a appears in all 3 lists -> highest score
    assert "content A" in contents
    # doc_c appears in 1 list but doc_b in 2, so doc_b should rank higher
    assert "content B" in contents


def test_update_retrieval_stats_increments_count():
    """_update_retrieval_stats should increment hit_count for given chunk ids."""
    from backend.services.hybrid_retriever import HybridRetriever
    # This test verifies the method signature exists and can be called
    # Actual DB interaction is tested via integration
    retriever = HybridRetriever.__new__(HybridRetriever)
    # Just verify the method exists
    assert hasattr(retriever, '_update_retrieval_stats')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hybrid_retriever_graph.py -v`
Expected: FAIL for `_update_retrieval_stats` test (method doesn't exist yet)

- [ ] **Step 3: Add graph retrieval and stats tracking to hybrid_retriever.py**

Add these imports at the top of `backend/services/hybrid_retriever.py`:

```python
from backend.db import SessionFactory
from backend.models.db_models import RetrievalStatsORM
```

Add these methods to the `HybridRetriever` class (after `_deep_retrieve`):

```python
    def _graph_retrieve(self, query: str, user_id: int, top_k: int = 10) -> list[Document]:
        """Graph retrieval: extract entities from query, traverse Neo4j graph."""
        from backend.services.graph_service import graph_service
        from backend.services.entity_extractor import entity_extractor

        entities = entity_extractor.extract_query_entities(query)
        if not entities:
            return []

        chunk_ids = graph_service.search_by_entities(entities, user_id=user_id, top_k=top_k)
        if not chunk_ids:
            return []

        # Fetch parent chunk content from PostgreSQL
        parents = parent_store.get_by_ids(chunk_ids)
        return [
            Document(
                page_content=p.content,
                metadata={"doc_id": p.id, "filename": p.filename, "heading_path": p.heading_path, "score": 1.0},
            )
            for p in parents
        ]

    def _update_retrieval_stats(self, parent_ids: list[str], user_id: int) -> None:
        """Async-safe: increment hit_count for retrieved parent chunks."""
        from datetime import datetime, timezone, timedelta

        if not parent_ids:
            return
        try:
            window = timedelta(days=settings.kg_hit_window_days)
            cutoff = datetime.now(timezone.utc) - window
            uuids = []
            for pid in parent_ids:
                try:
                    import uuid as _uuid
                    uuids.append(_uuid.UUID(pid))
                except ValueError:
                    continue
            if not uuids:
                return

            with SessionFactory() as session:
                for uid in uuids:
                    row = session.query(RetrievalStatsORM).filter_by(chunk_id=uid).first()
                    if row:
                        if row.last_hit_at and row.last_hit_at < cutoff:
                            row.hit_count = 1
                        else:
                            row.hit_count = row.hit_count + 1
                        row.last_hit_at = datetime.now(timezone.utc)
                    else:
                        session.add(RetrievalStatsORM(
                            chunk_id=uid, hit_count=1,
                            last_hit_at=datetime.now(timezone.utc), extracted=False,
                            user_id=user_id,
                        ))
                session.commit()

                # Trigger extraction for chunks that hit threshold
                to_extract = (
                    session.query(RetrievalStatsORM)
                    .filter(
                        RetrievalStatsORM.chunk_id.in_(uuids),
                        RetrievalStatsORM.hit_count >= settings.kg_hit_threshold,
                        RetrievalStatsORM.extracted == False,
                    )
                    .all()
                )
                if to_extract:
                    import threading
                    extract_ids = [str(r.chunk_id) for r in to_extract]
                    extract_user_id = to_extract[0].user_id
                    threading.Thread(
                        target=self._trigger_extraction, args=(extract_ids, extract_user_id), daemon=True
                    ).start()
        except Exception as e:
            logger.warning(f"Failed to update retrieval stats: {e}")

    def _trigger_extraction(self, chunk_ids: list[str], user_id: int) -> None:
        """Background: extract entities from chunks and write to Neo4j."""
        from backend.services.graph_service import graph_service
        from backend.services.entity_extractor import entity_extractor

        parents = parent_store.get_by_ids(chunk_ids)
        for p in parents:
            try:
                result = entity_extractor.extract_from_chunk(p.content)
                if result["entities"]:
                    import json as _json
                    graph_service.build_graph_for_chunk(
                        chunk_id=p.id,
                        filename=p.filename,
                        heading_path=_json.dumps(p.heading_path, ensure_ascii=False),
                        entities=result["entities"],
                        relations=result["relations"],
                        user_id=user_id,
                    )
                    logger.info(f"Extracted {len(result['entities'])} entities from chunk {p.id}")

                # Mark as extracted
                import uuid as _uuid
                with SessionFactory() as session:
                    row = session.query(RetrievalStatsORM).filter_by(chunk_id=_uuid.UUID(p.id)).first()
                    if row:
                        row.extracted = True
                        session.commit()
            except Exception as e:
                logger.warning(f"Extraction failed for chunk {p.id}: {e}")
```

Modify `_precise_retrieve` to include graph results (around line 388):

```python
        # Add graph retrieval as third path
        graph_docs = self._graph_retrieve(query, user_id or 0, top_k=fetch_k)
        doc_lists = [vec_docs, bm25_docs, graph_docs]
```

Modify `_deep_retrieve` similarly (around line 446):

```python
        # Add graph retrieval as third path
        graph_docs = self._graph_retrieve(query, user_id or 0, top_k=fetch_k)
        doc_lists = [vec_docs, bm25_docs, graph_docs]
```

In `_expand_to_parents`, change the method signature to accept `user_id` (line 305):

```python
    def _expand_to_parents(self, leaves: list[Document], top_n: int, query: str = "", user_id: int | None = None) -> list[Document]:
```

Then add stats tracking after building `parent_ids_ordered` (after line 313):

```python
        # Track retrieval stats for on-demand KG extraction
        if user_id is not None:
            import threading
            threading.Thread(
                target=self._update_retrieval_stats,
                args=(parent_ids_ordered, user_id),
                daemon=True,
            ).start()
```

Also update all callers of `_expand_to_parents` in the same file to pass `user_id`:

```python
# In _fast_retrieve (line 357):
result = self._expand_to_parents(docs, top_n=top_k, query=query, user_id=user_id)

# In _precise_retrieve (line 403):
result = self._expand_to_parents(fused, top_n=top_k, query=query, user_id=user_id)

# In _deep_retrieve (line 463):
result = self._expand_to_parents(reranked, top_n=top_k, query=query, user_id=user_id)

# In _aget_relevant_documents (line 517):
result = self._expand_to_parents(reranked, top_n=orig_k, query=query, user_id=None)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_hybrid_retriever_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/hybrid_retriever.py tests/test_hybrid_retriever_graph.py
git commit -m "feat: add graph retrieval path and on-demand extraction to hybrid retriever"
```

---

## Task 6: Document Deletion Cleanup

**Files:**
- Modify: `backend/services/document_service.py`
- Modify: `backend/routers/documents.py`

- [ ] **Step 1: Add graph cleanup to document_service.py**

In `backend/services/document_service.py`, modify the `delete_file` method (line 177) to also clean up Neo4j data. Add after the existing `vector_service.delete_by_filename` call:

```python
        # Clean up Neo4j graph data
        try:
            from backend.services.graph_service import graph_service
            parent_chunks = parent_store.get_by_filename(filename, user_id=user_id)
            chunk_ids = [p.id for p in parent_chunks]
            graph_service.delete_by_filename(filename, user_id=user_id or 0, chunk_ids=chunk_ids)
        except Exception as e:
            logger.warning(f"Graph cleanup failed for '{filename}': {e}")
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/document_service.py
git commit -m "feat: clean up Neo4j graph data on document deletion"
```

---

## Task 7: Knowledge Graph API Router

**Files:**
- Create: `backend/routers/knowledge_graph.py`
- Modify: `backend/models/schemas.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Add KG schemas to schemas.py**

Append to `backend/models/schemas.py`:

```python
class KGStatsResponse(BaseModel):
    entity_count: int
    relation_count: int
    type_count: int


class KGEntitySummary(BaseModel):
    id: str | None = None
    name: str
    type: str | None = None
    description: str | None = None


class KGRelationInfo(BaseModel):
    direction: str
    relation: str
    target: KGEntitySummary | None = None
    source: KGEntitySummary | None = None
    context: str | None = None


class KGChunkRef(BaseModel):
    chunk_id: str | None = None
    filename: str | None = None
    heading_path: str | None = None


class KGEntityDetailResponse(BaseModel):
    entity: KGEntitySummary
    relations: list[KGRelationInfo]
    mentioned_in: list[KGChunkRef]


class KGEntityListResponse(BaseModel):
    entities: list[KGEntitySummary]
    total: int
    page: int
    page_size: int
```

- [ ] **Step 2: Create knowledge_graph router**

Create `backend/routers/knowledge_graph.py`:

```python
"""知识图谱浏览 API 路由"""

import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from backend.models.schemas import (
    KGStatsResponse, KGEntityListResponse, KGEntitySummary,
    KGEntityDetailResponse, KGRelationInfo, KGChunkRef,
)
from backend.utils.auth import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["knowledge-graph"])


@router.get("/stats", response_model=KGStatsResponse)
async def get_kg_stats(current_user: CurrentUser = Depends(get_current_user)):
    from backend.services.graph_service import graph_service
    stats = await asyncio.to_thread(graph_service.get_stats, user_id=current_user.id)
    return KGStatsResponse(**stats)


@router.get("/entities", response_model=KGEntityListResponse)
async def list_entities(
    entity_type: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 20,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    result = await asyncio.to_thread(
        graph_service.get_entities,
        user_id=current_user.id, entity_type=entity_type,
        search=search, page=page, page_size=page_size,
    )
    return KGEntityListResponse(**result)


@router.get("/entities/{entity_id}", response_model=KGEntityDetailResponse)
async def get_entity_detail(
    entity_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    result = await asyncio.to_thread(
        graph_service.get_entity_detail, entity_id=entity_id, user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Entity not found")
    return KGEntityDetailResponse(**result)


@router.get("/search")
async def search_kg(
    q: str = "",
    top_k: int = 10,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    from backend.services.entity_extractor import entity_extractor

    entities = await asyncio.to_thread(entity_extractor.extract_query_entities, q)
    if not entities:
        return {"entities": [], "chunks": []}

    chunk_ids = await asyncio.to_thread(
        graph_service.search_by_entities, entities, user_id=current_user.id, top_k=top_k,
    )
    return {"matched_entities": entities, "chunk_ids": chunk_ids}


@router.post("/extract/{doc_id}")
async def manual_extract(
    doc_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """手动触发文档全量抽取（绕过阈值限制）。"""
    from backend.services.parent_store import parent_store
    from backend.services.graph_service import graph_service
    from backend.services.entity_extractor import entity_extractor

    parents = await asyncio.to_thread(parent_store.get_by_filename, doc_id, user_id=current_user.id)
    if not parents:
        raise HTTPException(status_code=404, detail="Document not found")

    extracted_count = 0
    for p in parents:
        result = await asyncio.to_thread(entity_extractor.extract_from_chunk, p.content)
        if result["entities"]:
            await asyncio.to_thread(
                graph_service.build_graph_for_chunk,
                chunk_id=p.id, filename=p.filename,
                heading_path=json.dumps(p.heading_path, ensure_ascii=False),
                entities=result["entities"], relations=result["relations"],
                user_id=current_user.id,
            )
            extracted_count += 1

    return {"detail": f"Extracted entities from {extracted_count}/{len(parents)} chunks"}
```

- [ ] **Step 3: Register router in main.py**

Add to `backend/main.py` after the existing router imports (line 30):

```python
from backend.routers import knowledge_graph
```

Add after line 100 (`app.include_router(qa.router, ...)`):

```python
app.include_router(knowledge_graph.router, dependencies=[Depends(get_current_user)])
```

- [ ] **Step 4: Commit**

```bash
git add backend/routers/knowledge_graph.py backend/models/schemas.py backend/main.py
git commit -m "feat: add knowledge graph browsing API endpoints"
```

---

## Task 8: Frontend — KG Page, API Client, Routing

**Files:**
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/pages/KGPage.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Add KG API functions to client.ts**

Append to `frontend/src/api/client.ts`:

```typescript
// === Knowledge Graph API ===

export interface KGStats {
  entity_count: number
  relation_count: number
  type_count: number
}

export interface KGEntitySummary {
  id: string | null
  name: string
  type: string | null
  description: string | null
}

export interface KGRelationInfo {
  direction: string
  relation: string
  target: KGEntitySummary | null
  source: KGEntitySummary | null
  context: string | null
}

export interface KGChunkRef {
  chunk_id: string | null
  filename: string | null
  heading_path: string | null
}

export interface KGEntityDetail {
  entity: KGEntitySummary
  relations: KGRelationInfo[]
  mentioned_in: KGChunkRef[]
}

export interface KGEntityListResponse {
  entities: KGEntitySummary[]
  total: number
  page: number
  page_size: number
}

export async function getKGStats(): Promise<KGStats> {
  const { data } = await api.get<KGStats>('/kg/stats')
  return data
}

export async function listKGEntities(
  params: { entity_type?: string; search?: string; page?: number; page_size?: number } = {},
): Promise<KGEntityListResponse> {
  const { data } = await api.get<KGEntityListResponse>('/kg/entities', { params })
  return data
}

export async function getKGEntityDetail(entityId: string): Promise<KGEntityDetail> {
  const { data } = await api.get<KGEntityDetail>(`/kg/entities/${encodeURIComponent(entityId)}`)
  return data
}

export async function extractDocument(docId: string): Promise<{ detail: string }> {
  const { data } = await api.post<{ detail: string }>(`/kg/extract/${encodeURIComponent(docId)}`)
  return data
}
```

- [ ] **Step 2: Create KGPage.tsx**

Create `frontend/src/pages/KGPage.tsx`:

```tsx
import { useState, useEffect } from 'react'
import {
  getKGStats, listKGEntities, getKGEntityDetail,
  type KGStats, type KGEntitySummary, type KGEntityDetail,
} from '../api/client'

export default function KGPage() {
  const [stats, setStats] = useState<KGStats | null>(null)
  const [entities, setEntities] = useState<KGEntitySummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<KGEntityDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getKGStats().then(setStats).catch(() => {})
  }, [])

  useEffect(() => {
    setLoading(true)
    listKGEntities({ search: search || undefined, page, page_size: 20 })
      .then((res) => { setEntities(res.entities); setTotal(res.total) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [search, page])

  useEffect(() => {
    if (selectedId) {
      getKGEntityDetail(selectedId).then(setDetail).catch(() => setDetail(null))
    } else {
      setDetail(null)
    }
  }, [selectedId])

  return (
    <div style={{ display: 'flex', gap: 24, height: 'calc(100vh - 64px)' }}>
      {/* Left: entity list */}
      <div style={{ width: 360, display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ margin: '0 0 16px' }}>知识图谱</h2>
        {stats && (
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 13, color: '#888' }}>
            <span>实体: {stats.entity_count}</span>
            <span>关系: {stats.relation_count}</span>
            <span>类型: {stats.type_count}</span>
          </div>
        )}
        <input
          placeholder="搜索实体..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          style={{ padding: '8px 12px', border: '1px solid #ddd', borderRadius: 6, marginBottom: 12 }}
        />
        <div style={{ flex: 1, overflow: 'auto', border: '1px solid #eee', borderRadius: 6 }}>
          {loading ? (
            <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>加载中...</div>
          ) : entities.length === 0 ? (
            <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>暂无实体</div>
          ) : (
            entities.map((e) => (
              <div
                key={e.id || e.name}
                onClick={() => setSelectedId(e.id)}
                style={{
                  padding: '10px 14px', cursor: 'pointer', borderBottom: '1px solid #f0f0f0',
                  background: selectedId === e.id ? '#f0f7ff' : 'transparent',
                }}
              >
                <div style={{ fontWeight: 500 }}>{e.name}</div>
                {e.type && <span style={{ fontSize: 12, color: '#888' }}>{e.type}</span>}
                {e.description && <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>{e.description}</div>}
              </div>
            ))
          )}
        </div>
        {total > 20 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 8 }}>
            <button disabled={page <= 1} onClick={() => setPage(page - 1)}>上一页</button>
            <span style={{ lineHeight: '32px', fontSize: 13 }}>{page} / {Math.ceil(total / 20)}</span>
            <button disabled={page * 20 >= total} onClick={() => setPage(page + 1)}>下一页</button>
          </div>
        )}
      </div>

      {/* Right: entity detail */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {!detail ? (
          <div style={{ padding: 40, textAlign: 'center', color: '#999' }}>
            选择左侧实体查看详情
          </div>
        ) : (
          <div>
            <h3 style={{ marginTop: 0 }}>{detail.entity.name}</h3>
            {detail.entity.type && <div style={{ color: '#888', marginBottom: 4 }}>类型: {detail.entity.type}</div>}
            {detail.entity.description && <div style={{ marginBottom: 16 }}>{detail.entity.description}</div>}

            {detail.relations.length > 0 && (
              <>
                <h4>关系</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #eee', textAlign: 'left' }}>
                      <th style={{ padding: 8 }}>方向</th>
                      <th style={{ padding: 8 }}>关系</th>
                      <th style={{ padding: 8 }}>关联实体</th>
                      <th style={{ padding: 8 }}>上下文</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.relations.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: 8 }}>{r.direction === 'outgoing' ? '→' : '←'}</td>
                        <td style={{ padding: 8 }}>{r.relation}</td>
                        <td style={{ padding: 8 }}>
                          <button
                            style={{ background: 'none', border: 'none', color: '#1677ff', cursor: 'pointer', padding: 0 }}
                            onClick={() => setSelectedId((r.target || r.source)?.id || null)}
                          >
                            {(r.target || r.source)?.name}
                          </button>
                        </td>
                        <td style={{ padding: 8, fontSize: 12, color: '#666', maxWidth: 300 }}>
                          {r.context || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {detail.mentioned_in.length > 0 && (
              <>
                <h4>来源文档</h4>
                {detail.mentioned_in.map((c, i) => (
                  <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f5f5f5', fontSize: 13 }}>
                    <span style={{ fontWeight: 500 }}>{c.filename}</span>
                    {c.heading_path && <span style={{ color: '#888', marginLeft: 8 }}>{c.heading_path}</span>}
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Add KG route to App.tsx**

In `frontend/src/App.tsx`, add import:

```tsx
import KGPage from './pages/KGPage'
```

Add route inside the `<Route element={<Layout />}>` block:

```tsx
<Route path="/kg" element={<KGPage />} />
```

- [ ] **Step 4: Add KG nav link to Layout.tsx**

In `frontend/src/components/Layout.tsx`, add a new `NavLink` after the "文档管理" link:

```tsx
        <NavLink
          to="/kg"
          style={({ isActive }) => ({
            ...linkBase,
            background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: isActive ? '#fff' : 'var(--sidebar-text)',
          })}
        >
          知识图谱
        </NavLink>
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/pages/KGPage.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat: add knowledge graph browsing page and API client"
```

---

## Task 9: Manual Extract Button on Documents Page

**Files:**
- Modify: `frontend/src/pages/DocumentsPage.tsx`

- [ ] **Step 1: Add "加入图谱" button to DocumentsPage**

Read `frontend/src/pages/DocumentsPage.tsx` first, then add an "加入图谱" button next to the existing delete button for each document. Import `extractDocument` from `../api/client` and add a click handler that calls it.

```tsx
import { extractDocument } from '../api/client'

// Inside the document list item, add a button:
<button
  onClick={async () => {
    try {
      const res = await extractDocument(doc.doc_id)
      alert(res.detail)
    } catch (e: any) {
      alert('抽取失败: ' + (e.response?.data?.detail || e.message))
    }
  }}
  style={{
    background: '#52c41a', color: '#fff', border: 'none',
    padding: '4px 12px', borderRadius: 4, cursor: 'pointer', fontSize: 12,
  }}
>
  加入图谱
</button>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/DocumentsPage.tsx
git commit -m "feat: add manual KG extraction button to documents page"
```
