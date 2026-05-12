import os

# HuggingFace 镜像必须在任何模型加载前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类
    使用 pydantic-settings 从环境变量和 .env 文件读取配置
    """

    # ==================== Qwen LLM 配置 ====================
    qwen_api_key: str  # 阿里云 DashScope API 密钥（必填）
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API 地址
    qwen_model: str = "qwen-plus"  # 使用的 Qwen 模型
    qwen_max_tokens: int = 8192  # LLM 最大输出 token 数

    # ==================== Embedding 配置 ====================
    embedding_model: str = "BAAI/bge-small-zh-v1.5"  # Embedding 模型名称
    embedding_device: str = "cpu"  # Embedding 模型运行设备

    # ==================== Reranker 配置 ====================
    reranker_model: str = "BAAI/bge-reranker-base"  # 重排序模型名称

    # ==================== ChromaDB 配置 ====================
    chroma_persist_dir: str = "data/chroma_db"  # 向量数据库持久化目录
    chroma_collection: str = "knowledge_base"    # 向量集合名称

    # ==================== 文档分块配置 ====================
    chunk_size: int = 500      # 分块大小（字符数）
    chunk_overlap: int = 50   # 分块重叠大小

    # === PostgreSQL ===
    postgres_url: str = "postgresql+psycopg2://knowrag:knowrag@localhost:5432/knowrag"
    pg_pool_size: int = 5
    pg_pool_max_overflow: int = 10
    auto_migrate: bool = True

    # === Chunking ===
    parent_max_chars: int = 1500
    leaf_chunk_size: int = 300
    leaf_chunk_overlap: int = 30

    # === PDF heuristic thresholds ===
    pdf_h1_ratio: float = 1.4
    pdf_h2_ratio: float = 1.2
    pdf_h3_ratio: float = 1.05

    # ==================== 文件上传配置 ====================
    upload_dir: str = "data/uploads"  # 上传文件存储目录
    max_upload_size_mb: int = 50      # 最大上传文件大小（MB）

    # ==================== HuggingFace 配置 ====================
    hf_endpoint: str = "https://hf-mirror.com"  # HuggingFace 镜像地址

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例（带缓存）

    Returns:
        Settings 实例
    """
    return Settings()
