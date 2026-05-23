"""
一次性脚本：将 ChromaDB 中所有叶子块迁移到 Milvus。

使用方法：
    python scripts/migrate_chroma_to_milvus.py

前置条件：
    - ChromaDB 持久化目录存在（data/chroma_db/）
    - Milvus 容器已启动（docker-compose up -d etcd milvus）
    - Milvus collection 已由 VectorService 自动创建（需先导入 vector_service）
"""

import json
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb import PersistentClient
from backend.config import get_settings
from backend.services.embedding_service import embedding_service
from backend.services.vector_service import vector_service

settings = get_settings()


def migrate():
    # 1. 连接 ChromaDB
    chroma_client = PersistentClient(path=settings.chroma_persist_dir)
    chroma_col = chroma_client.get_collection(settings.chroma_collection)

    total = chroma_col.count()
    print(f"ChromaDB collection '{settings.chroma_collection}': {total} chunks")

    if total == 0:
        print("Nothing to migrate.")
        return

    # 2. 检查 Milvus 是否已有数据，避免重复迁移
    existing = vector_service.count()
    if existing > 0:
        print(f"Milvus already has {existing} chunks. "
              f"Drop the collection or clear data before re-migrating.")
        return

    # 3. 分批读取 ChromaDB 并写入 Milvus
    batch_size = 100
    offset = 0

    while offset < total:
        results = chroma_col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            break

        ids = results["ids"]
        texts = results["documents"] or [""] * len(ids)
        metas = results["metadatas"] or [{}] * len(ids)

        # 重新计算 embedding
        embeddings = embedding_service.embed(texts)

        rows = []
        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            heading = meta.get("heading_path")
            if heading is None:
                heading_json = meta.get("heading_path_json", "[]")
                try:
                    heading = json.loads(heading_json) if isinstance(heading_json, str) else heading_json
                except (json.JSONDecodeError, TypeError):
                    heading = []
            rows.append({
                "id": ids[i],
                "content": texts[i] if i < len(texts) else "",
                "embedding": embeddings[i],
                "parent_id": str(meta.get("parent_id", "")),
                "filename": str(meta.get("filename", "")),
                "heading_path_json": json.dumps(
                    heading if isinstance(heading, list) else [], ensure_ascii=False
                ),
                "page": int(meta.get("page", 0) or 0),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "preserve": bool(meta.get("preserve", False)),
            })

        col = vector_service.collection
        col.insert(rows)
        col.flush()

        offset += len(ids)
        print(f"Migrated {min(offset, total)}/{total} chunks")

    # 4. 创建索引
    vector_service._ensure_index()
    vector_service.create_indexes()

    print(f"\nMigration complete. Milvus now has {vector_service.count()} chunks.")
    print(f"ChromaDB data at '{settings.chroma_persist_dir}' can be archived.")


if __name__ == "__main__":
    migrate()
