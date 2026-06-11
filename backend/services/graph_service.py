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
import uuid as _uuid
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
        """为一个父块构建图谱节点和关系。使用 MERGE 语义：同名实体自动合并，关系追加。"""
        with self._driver.session() as session:
            # 1. MERGE parent chunk node
            session.run(
                "MERGE (c:ParentChunk {id: $chunk_id}) "
                "SET c.filename = $filename, c.heading_path = $heading_path, c.user_id = $user_id",
                chunk_id=chunk_id, filename=filename,
                heading_path=heading_path, user_id=user_id,
            )

            # 2. MERGE entity nodes + MENTIONED_IN relationships
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
        """根据实体名称进行图谱检索，返回关联的 ParentChunk id 列表。"""
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
            if chunk_ids:
                session.run(
                    "MATCH ()-[r:RELATES_TO]->() "
                    "WHERE r.chunk_id IN $chunk_ids "
                    "DELETE r",
                    chunk_ids=chunk_ids,
                )

            session.run(
                "MATCH (c:ParentChunk {filename: $filename, user_id: $user_id}) "
                "DETACH DELETE c",
                filename=filename, user_id=user_id,
            )

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
