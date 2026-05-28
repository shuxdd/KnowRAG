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

from pymilvus import MilvusClient, DataType
from langchain_core.documents import Document
from backend.config import get_settings
from backend.models.chunk_types import LeafChunk
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION_NAME = settings.milvus_collection
EMBEDDING_DIM = 512
MAX_STRING_LEN = 512


USER_ID_FIELD = "user_id"


def _ensure_client() -> MilvusClient:
    """连接 Milvus 并确保集合存在，返回 MilvusClient 实例。"""
    try:
        client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
        )

        if not client.has_collection(COLLECTION_NAME):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR,
                             is_primary=True, max_length=64)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR,
                             max_length=65535)
            schema.add_field(field_name="embedding",
                             datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR,
                             max_length=64)
            schema.add_field(field_name="filename", datatype=DataType.VARCHAR,
                             max_length=MAX_STRING_LEN)
            schema.add_field(field_name="heading_path_json",
                             datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="page", datatype=DataType.INT64)
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(field_name="preserve", datatype=DataType.BOOL)
            schema.add_field(field_name=USER_ID_FIELD, datatype=DataType.INT64)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="HNSW",
                params={"M": 16, "efConstruction": 200},
            )

            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=index_params,
            )
            logger.info(
                "Created collection '%s' with HNSW + COSINE index",
                COLLECTION_NAME,
            )
        return client
    except Exception:
        logger.error(
            "Failed to connect to Milvus at %s:%s",
            settings.milvus_host, settings.milvus_port, exc_info=True,
        )
        raise


_client_instance: MilvusClient | None = None


def _get_client() -> MilvusClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = _ensure_client()
    return _client_instance


class VectorService:
    """基于 MilvusClient 的向量检索服务。"""

    # ==================== 数据写入 ====================

    def add_documents(self, docs: List[Document], user_id: int = 0) -> List[str]:
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
                USER_ID_FIELD: user_id,
            })

        client = _get_client()
        client.insert(COLLECTION_NAME, rows)
        client.flush(COLLECTION_NAME)
        return ids

    def add_leaves(self, leaves: list[LeafChunk], user_id: int = 0) -> list[str]:
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
                USER_ID_FIELD: user_id,
            })

        client = _get_client()
        client.insert(COLLECTION_NAME, rows)
        client.flush(COLLECTION_NAME)
        self._ensure_index()
        return ids

    # ==================== 数据检索 ====================

    def similarity_search(self, query: str, k: int = 10, user_id: int | None = None) -> List[Document]:
        try:
            client = _get_client()
            query_vec = embedding_service.embed_query(query)

            filter_expr = ""
            if user_id is not None:
                filter_expr = f"{USER_ID_FIELD} == {user_id}"

            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=k,
                filter=filter_expr or None,
                output_fields=[
                    "id", "content", "filename", "parent_id",
                    "heading_path_json", "page", "chunk_index", "preserve",
                    USER_ID_FIELD,
                ],
            )

            docs = []
            if results and results[0]:
                for hit in results[0]:
                    score = max(0.0, min(1.0, hit.get("distance", 0)))
                    metadata = {
                        "parent_id": hit.get("parent_id", ""),
                        "filename": hit.get("filename", ""),
                        "heading_path": json.loads(
                            hit.get("heading_path_json", "[]")
                        ),
                        "page": hit.get("page", 0),
                        "chunk_index": hit.get("chunk_index", 0),
                        "preserve": hit.get("preserve", False),
                        "doc_id": hit.get("id", ""),
                        "score": score,
                    }
                    docs.append(Document(
                        page_content=hit.get("content", ""),
                        metadata=metadata,
                    ))
            return docs
        except Exception:
            logger.warning("similarity_search failed", exc_info=True)
            return []

    def query_with_filter(
        self, query: str, where: dict, n_results: int = 5, user_id: int | None = None
    ) -> dict:
        try:
            client = _get_client()
            query_vec = embedding_service.embed_query(query)
            filter_expr = self._build_filter_expr(where, user_id=user_id)

            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=n_results,
                filter=filter_expr,
                output_fields=[
                    "id", "content", "filename", "parent_id",
                    "heading_path_json", "page", "chunk_index", "preserve",
                    USER_ID_FIELD,
                ],
            )

            ids_list = []
            docs_list = []
            metas_list = []
            distances_list = []

            if results and results[0]:
                for hit in results[0]:
                    ids_list.append(hit.get("id", ""))
                    docs_list.append(hit.get("content", ""))
                    metas_list.append({
                        "parent_id": hit.get("parent_id", ""),
                        "filename": hit.get("filename", ""),
                        "heading_path": json.loads(
                            hit.get("heading_path_json", "[]")
                        ),
                        "page": hit.get("page", 0),
                        "chunk_index": hit.get("chunk_index", 0),
                        "preserve": hit.get("preserve", False),
                        "user_id": hit.get(USER_ID_FIELD, 0),
                    })
                    distances_list.append(
                        max(0.0, min(1.0, 1.0 - hit.get("distance", 0)))
                    )

            return {
                "ids": [ids_list],
                "documents": [docs_list],
                "metadatas": [metas_list],
                "distances": [distances_list],
            }
        except Exception:
            logger.warning("query_with_filter failed", exc_info=True)
            return {
                "ids": [[]], "documents": [[]],
                "metadatas": [[]], "distances": [[]],
            }

    # ==================== 数据查询 ====================

    def get_by_filename(self, filename: str, user_id: int | None = None) -> list[dict]:
        client = _get_client()
        filt = f'filename == "{self._escape_str(filename)}"'
        if user_id is not None:
            filt += f" && {USER_ID_FIELD} == {user_id}"
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=filt,
            output_fields=[
                "id", "content", "filename", "parent_id",
                "heading_path_json", "page", "chunk_index", "preserve",
                USER_ID_FIELD,
            ],
        )
        return [
            {
                "id": r.get("id", ""),
                "metadata": {
                    "parent_id": r.get("parent_id", ""),
                    "filename": r.get("filename", ""),
                    "heading_path": json.loads(
                        r.get("heading_path_json", "[]")
                    ),
                    "page": r.get("page", 0),
                    "chunk_index": r.get("chunk_index", 0),
                    "preserve": r.get("preserve", False),
                    "user_id": r.get(USER_ID_FIELD, 0),
                },
                "document": r.get("content", ""),
            }
            for r in results
        ]

    def get_by_parent_id(self, parent_id: str) -> list[dict]:
        client = _get_client()
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'parent_id == "{self._escape_str(parent_id)}"',
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
                    "heading_path": json.loads(
                        r.get("heading_path_json", "[]")
                    ),
                    "page": r.get("page", 0),
                    "chunk_index": r.get("chunk_index", 0),
                    "preserve": r.get("preserve", False),
                },
                "document": r.get("content", ""),
            }
            for r in results
        ]

    def get_document_stats(self, user_id: int | None = None) -> List[dict]:
        client = _get_client()
        try:
            filt = f"{USER_ID_FIELD} == {user_id}" if user_id is not None else "id != ''"
            results = client.query(
                collection_name=COLLECTION_NAME,
                filter=filt,
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

    def get_all_chunks(self, user_id: int | None = None) -> List[Document]:
        client = _get_client()
        try:
            filt = f"{USER_ID_FIELD} == {user_id}" if user_id is not None else "id != ''"
            results = client.query(
                collection_name=COLLECTION_NAME,
                filter=filt,
                output_fields=[
                    "id", "content", "filename", "parent_id",
                    "heading_path_json", "page", "chunk_index", "preserve",
                    USER_ID_FIELD,
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
                    "user_id": r.get(USER_ID_FIELD, 0),
                },
            ))
        return docs

    def count(self) -> int:
        client = _get_client()
        try:
            stats = client.get_collection_stats(COLLECTION_NAME)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    # ==================== 数据删除 ====================

    def delete_by_filename(self, filename: str, user_id: int | None = None) -> int:
        existing = self.get_by_filename(filename, user_id=user_id)
        count = len(existing)
        if count == 0:
            return 0

        client = _get_client()
        filt = f'filename == "{self._escape_str(filename)}"'
        if user_id is not None:
            filt += f" && {USER_ID_FIELD} == {user_id}"
        client.delete(
            collection_name=COLLECTION_NAME,
            filter=filt,
        )
        client.flush(COLLECTION_NAME)
        return count

    # ==================== 内部方法 ====================

    @staticmethod
    def _escape_str(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _build_filter_expr(self, where: dict, user_id: int | None = None) -> str:
        parts = []
        if user_id is not None:
            parts.append(f"{USER_ID_FIELD} == {user_id}")
        for key, value in where.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{self._escape_str(value)}"')
            elif isinstance(value, bool):
                parts.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {value}")
        return " && ".join(parts) if parts else "id != ''"

    def _ensure_index(self):
        client = _get_client()
        if client.list_indexes(COLLECTION_NAME, field_name="embedding"):
            return
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index(COLLECTION_NAME, index_params)

    def create_indexes(self):
        client = _get_client()
        for field_name in ("filename", "parent_id"):
            try:
                index_params = client.prepare_index_params()
                index_params.add_index(field_name=field_name)
                client.create_index(
                    collection_name=COLLECTION_NAME,
                    index_params=index_params,
                )
            except Exception:
                logger.warning(
                    f"Failed to create index on {field_name}", exc_info=True,
                )


# 全局单例
vector_service = VectorService()
