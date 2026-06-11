"""
独立的 embedding 服务模块

将 SentenceTransformer 的加载和 encode 从 VectorService 中解耦，
使 VectorService 可被 ChromaDB 和 Milvus 共用。

模型：由 config.embedding_model 配置决定
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
from sentence_transformers import SentenceTransformer
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """SentenceTransformer embedding 封装，单例模式。"""

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        self._dim = self._model.get_embedding_dimension()

    @property
    def dim(self) -> int:
        """Embedding 向量维度。"""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表。"""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """将单条查询文本转为向量。"""
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# 全局单例
embedding_service = EmbeddingService()
