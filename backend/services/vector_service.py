"""
向量检索服务模块

基于 Milvus 实现向量存储和相似度检索。
使用 Sentence Transformer (bge-small-zh-v1.5) 将文本转换为向量。

核心功能：
- add_leaves(): 添加叶子块到向量库
- similarity_search(): 向量相似度检索
- delete_by_filename(): 按文件名删除文档
- get_document_stats(): 获取文档统计信息
- get_all_chunks(): 获取所有文档块（用于 BM25 重建）
- count(): 获取文档块总数
- get_by_filename(): 按文件名获取叶子块
- get_by_parent_id(): 按父块 ID 获取叶子块
- query_with_filter(): 带元数据过滤的语义检索

向量集合配置：
- 名称：knowledge_base（可配置）
- 距离度量：COSINE
- 向量维度：512（bge-small-zh-v1.5）
"""

import json
import logging
from typing import List

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from langchain_core.documents import Document
from backend.config import get_settings
from backend.models.chunk_types import LeafChunk
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION_NAME = settings.milvus_collection
EMBEDDING_DIM = 512
MAX_STRING_LEN = 512

FIELDS = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=MAX_STRING_LEN),
    FieldSchema(name="heading_path_json", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="preserve", dtype=DataType.BOOL),
]


def _ensure_collection() -> Collection:
    """连接 Milvus 并确保集合存在，返回已加载的 Collection 对象。"""
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )

    if utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
    else:
        schema = CollectionSchema(
            fields=FIELDS,
            description="KnowRAG knowledge base leaf chunks",
        )
        col = Collection(COLLECTION_NAME, schema=schema)

    col.load()
    return col


class VectorService:
    """基于 Milvus 的向量检索服务。"""

    def __init__(self):
        self._col: Collection | None = None

    @property
    def collection(self) -> Collection:
        """延迟初始化 Milvus 连接。"""
        if self._col is None:
            self._col = _ensure_collection()
        return self._col

    # ==================== 数据写入 ====================

    def add_documents(self, docs: List[Document]) -> List[str]:
        import uuid

        ids = [str(uuid.uuid4()) for _ in docs]
        texts = [doc.page_content for doc in docs]
        embeddings = embedding_service.embed(texts)

        rows = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            rows.append({
                "id": ids[i],
                "content": texts[i],
                "embedding": embeddings[i],
                "parent_id": str(meta.get("parent_id", "")),
                "filename": str(meta.get("filename", "")),
                "heading_path_json": json.dumps(
                    meta.get("heading_path", []), ensure_ascii=False
                ),
                "page": int(meta.get("page", 0) or 0),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "preserve": bool(meta.get("preserve", False)),
            })

        col = self.collection
        col.insert(rows)
        col.flush()
        return ids

    def add_leaves(self, leaves: list[LeafChunk]) -> list[str]:
        if not leaves:
            return []

        ids = [leaf.id for leaf in leaves]
        texts = [leaf.content for leaf in leaves]
        embeddings = embedding_service.embed(texts)

        rows = []
        for leaf in leaves:
            rows.append({
                "id": leaf.id,
                "content": leaf.content,
                "embedding": embeddings[len(rows)],
                "parent_id": leaf.parent_id,
                "filename": leaf.filename,
                "heading_path_json": json.dumps(
                    leaf.heading_path, ensure_ascii=False
                ),
                "page": leaf.page if leaf.page is not None else 0,
                "chunk_index": leaf.chunk_index,
                "preserve": leaf.preserve,
            })

        col = self.collection
        col.insert(rows)
        col.flush()
        self._ensure_index()
        return ids

    # ==================== 数据检索 ====================

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        col = self.collection
        query_vec = embedding_service.embed_query(query)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = col.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=[
                "id", "content", "filename", "parent_id",
                "heading_path_json", "page", "chunk_index", "preserve",
            ],
        )

        docs = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.entity
                score = max(0.0, min(1.0, hit.score))
                metadata = {
                    "parent_id": entity.get("parent_id", ""),
                    "filename": entity.get("filename", ""),
                    "heading_path": json.loads(
                        entity.get("heading_path_json", "[]")
                    ),
                    "page": entity.get("page", 0),
                    "chunk_index": entity.get("chunk_index", 0),
                    "preserve": entity.get("preserve", False),
                    "doc_id": entity.get("id", ""),
                    "score": score,
                }
                docs.append(Document(
                    page_content=entity.get("content", ""),
                    metadata=metadata,
                ))
        return docs

    def query_with_filter(
        self, query: str, where: dict, n_results: int = 5
    ) -> dict:
        col = self.collection
        query_vec = embedding_service.embed_query(query)
        filter_expr = self._build_filter_expr(where)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = col.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            expr=filter_expr,
            output_fields=[
                "id", "content", "filename", "parent_id",
                "heading_path_json", "page", "chunk_index", "preserve",
            ],
        )

        ids_list = []
        docs_list = []
        metas_list = []
        distances_list = []

        if results and results[0]:
            for hit in results[0]:
                entity = hit.entity
                ids_list.append(entity.get("id", ""))
                docs_list.append(entity.get("content", ""))
                metas_list.append({
                    "parent_id": entity.get("parent_id", ""),
                    "filename": entity.get("filename", ""),
                    "heading_path_json": entity.get("heading_path_json", "[]"),
                    "page": entity.get("page", 0),
                    "chunk_index": entity.get("chunk_index", 0),
                    "preserve": entity.get("preserve", False),
                })
                distances_list.append(1.0 - hit.score)

        return {
            "ids": [ids_list],
            "documents": [docs_list],
            "metadatas": [metas_list],
            "distances": [distances_list],
        }

    # ==================== 数据查询 ====================

    def get_by_filename(self, filename: str) -> list[dict]:
        col = self.collection
        results = col.query(
            expr=f'filename == "{filename}"',
            output_fields=[
                "id", "content", "filename", "parent_id",
                "heading_path_json", "page", "chunk_index", "preserve",
            ],
        )
        return [
            {
                "id": r.get("id", ""),
                "metadata": {
                    "parent_id": r.get("parent_id", ""),
                    "filename": r.get("filename", ""),
                    "heading_path_json": r.get("heading_path_json", "[]"),
                    "page": r.get("page", 0),
                    "chunk_index": r.get("chunk_index", 0),
                    "preserve": r.get("preserve", False),
                },
                "document": r.get("content", ""),
            }
            for r in results
        ]

    def get_by_parent_id(self, parent_id: str) -> list[dict]:
        col = self.collection
        results = col.query(
            expr=f'parent_id == "{parent_id}"',
            output_fields=[
                "id", "content", "filename", "parent_id",
                "heading_path_json", "page", "chunk_index", "preserve",
            ],
        )
        return [
            {
                "id": r.get("id", ""),
                "metadata": {
                    "parent_id": r.get("parent_id", ""),
                    "filename": r.get("filename", ""),
                    "heading_path_json": r.get("heading_path_json", "[]"),
                    "page": r.get("page", 0),
                    "chunk_index": r.get("chunk_index", 0),
                    "preserve": r.get("preserve", False),
                },
                "document": r.get("content", ""),
            }
            for r in results
        ]

    def get_document_stats(self) -> List[dict]:
        col = self.collection
        try:
            results = col.query(
                expr="id != ''",
                output_fields=["filename"],
            )
        except Exception:
            return []

        stats: dict[str, int] = {}
        for r in results:
            fn = r.get("filename", "unknown")
            stats[fn] = stats.get(fn, 0) + 1
        return [
            {"filename": fn, "chunks_count": cnt}
            for fn, cnt in stats.items()
        ]

    def get_all_chunks(self) -> List[Document]:
        col = self.collection
        try:
            results = col.query(
                expr="id != ''",
                output_fields=[
                    "id", "content", "filename", "parent_id",
                    "heading_path_json", "page", "chunk_index", "preserve",
                ],
            )
        except Exception:
            return []

        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.get("content", ""),
                metadata={
                    "parent_id": r.get("parent_id", ""),
                    "filename": r.get("filename", ""),
                    "heading_path": json.loads(r.get("heading_path_json", "[]")),
                    "page": r.get("page", 0),
                    "chunk_index": r.get("chunk_index", 0),
                    "preserve": r.get("preserve", False),
                },
            ))
        return docs

    def count(self) -> int:
        col = self.collection
        try:
            return col.num_entities
        except Exception:
            return 0

    # ==================== 数据删除 ====================

    def delete_by_filename(self, filename: str) -> int:
        existing = self.get_by_filename(filename)
        count = len(existing)
        if count == 0:
            return 0

        col = self.collection
        expr = f'filename == "{filename}"'
        col.delete(expr)
        col.flush()
        return count

    # ==================== 内部方法 ====================

    def _build_filter_expr(self, where: dict) -> str:
        parts = []
        for key, value in where.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                parts.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {value}")
        return " && ".join(parts) if parts else "id != ''"

    def _ensure_index(self):
        col = self.collection
        if not col.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            col.create_index("embedding", index_params)
            logger.info(
                f"Created IVF_FLAT index with COSINE metric on '{COLLECTION_NAME}'"
            )

    def create_indexes(self):
        col = self.collection
        for field_name in ("filename", "parent_id"):
            try:
                col.create_index(
                    field_name=field_name,
                    index_name=f"idx_{field_name}",
                )
            except Exception:
                pass


# 全局单例
vector_service = VectorService()
