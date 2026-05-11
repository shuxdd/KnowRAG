import os

# 必须在 chromadb 导入前设置，因为 chromadb 会触发 huggingface_hub 读环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uuid
from typing import List
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from backend.config import get_settings

settings = get_settings()


class VectorService:
    """
    向量检索服务
    基于 ChromaDB 实现向量存储和相似度检索
    使用 Sentence Transformer  embedding 模型将文本转换为向量
    """

    def __init__(self):
        """
        初始化向量服务
        - 创建 ChromaDB 持久化客户端
        - 初始化 embedding 函数
        - 获取或创建向量集合
        """
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self.client = PersistentClient(path=settings.chroma_persist_dir)
        # 使用 Sentence Transformer  embedding 模型
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        # 获取或创建名为 knowledge_base 的向量集合
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # 使用余弦距离
        )

    def add_documents(self, docs: List[Document]) -> List[str]:
        """
        向向量数据库添加文档

        Args:
            docs: LangChain Document 对象列表，每个 Document 包含 page_content 和 metadata

        Returns:
            返回新增文档的 ID 列表
        """
        ids = [str(uuid.uuid4()) for _ in docs]
        self.collection.add(
            ids=ids,
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )
        return ids

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """
        基于向量相似度进行文档检索

        Args:
            query: 查询文本
            k: 返回的最相似文档数量，默认为10

        Returns:
            返回 Document 对象列表，按相似度降序排列
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # 将距离转换为相似度分数（距离越小，相似度越高）
                score = 1.0 - (distance / 2.0)
                docs.append(
                    Document(
                        page_content=results["documents"][0][i],
                        metadata={
                            **metadata,
                            "doc_id": doc_id,
                            "score": max(0.0, min(1.0, score)),  # 确保分数在 [0, 1] 范围内
                        },
                    )
                )
        return docs

    def delete_by_filename(self, filename: str) -> int:
        """
        根据文件名删除向量数据库中的所有相关文档

        Args:
            filename: 要删除的文件名

        Returns:
            返回删除的文档块数量
        """
        results = self.collection.get(where={"filename": filename})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def get_document_stats(self) -> List[dict]:
        """
        获取已存储文档的统计信息

        Returns:
            返回文档统计列表，每个元素包含 filename 和 chunks_count（分块数量）
        """
        results = self.collection.get()
        if not results["metadatas"]:
            return []
        stats = {}
        for meta in results["metadatas"]:
            fn = meta.get("filename", "unknown")
            if fn not in stats:
                stats[fn] = {"filename": fn, "chunks_count": 0}
            stats[fn]["chunks_count"] += 1
        return list(stats.values())

    def get_all_chunks(self) -> List[Document]:
        """
        获取向量数据库中的所有文档块

        Returns:
            返回所有 Document 对象的列表
        """
        results = self.collection.get()
        if not results["ids"]:
            return []
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append(
                Document(
                    page_content=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )
        return docs

    def count(self) -> int:
        """
        获取向量数据库中的文档总数

        Returns:
            返回文档块的总数量
        """
        return self.collection.count()


# 全局单例实例
vector_service = VectorService()
