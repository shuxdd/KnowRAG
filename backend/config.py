"""
应用配置模块

本模块定义应用的所有配置项，包括：
- LLM（通义千问）配置
- Embedding 模型配置
- Reranker 模型配置
- ChromaDB 向量数据库配置
- PostgreSQL 数据库配置
- Redis 缓存配置
- 文档分块配置
- 文件上传配置
- JWT 认证配置

配置通过 pydantic-settings 从环境变量和 .env 文件读取。
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类
    使用 pydantic-settings 从环境变量和 .env 文件读取配置
    所有配置项都有默认值，可以在 .env 文件中覆盖
    """

    # ==================== Mimo LLM 配置 ====================
    mimo_api_key: str  # Mimo API 密钥（必填）
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"  # API 地址
    mimo_model: str = "mimo-v2.5"  # 使用的 Mimo 模型
    mimo_max_tokens: int = 8192  # LLM 最大输出 token 数

    # ==================== Embedding 配置 ====================
    embedding_model: str = "BAAI/bge-large-zh-v1.5"  # Embedding 模型名称
    embedding_device: str = "cpu"  # Embedding 模型运行设备

    # ==================== Reranker 配置 ====================
    reranker_model: str = "BAAI/bge-reranker-base"  # 重排序模型名称
    reranker_device: str = "cuda"  # 重排序模型运行设备（cuda/cpu）

    # ==================== ChromaDB 配置 ====================
    chroma_persist_dir: str = "data/chroma_db"  # 向量数据库持久化目录
    chroma_collection: str = "knowledge_base"    # 向量集合名称

    # ==================== Milvus 配置 ====================
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"
    milvus_search_ef: int = 100

    # ==================== 文档分块配置 ====================
    chunk_size: int = 500      # 分块大小（字符数）
    chunk_overlap: int = 50   # 分块重叠大小

    # === PostgreSQL ===
    postgres_url: str = "postgresql+psycopg://knowrag:knowrag@localhost:5433/knowrag?connect_timeout=5"
    pg_pool_size: int = 5
    pg_pool_max_overflow: int = 10
    pg_connect_timeout: int = 5
    auto_migrate: bool = True
    preload_models: bool = True

    # === Redis ===
    redis_url: str = "redis://localhost:6380/0"
    retrieval_cache_ttl: int = 600  # retrieval cache TTL in seconds

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

    # === Logging ===
    log_level: str = "INFO"

    # === Chunking ===
    parent_max_chars: int = 1500
    leaf_chunk_size: int = 300
    leaf_chunk_overlap: int = 30
    max_leaf_chars: int = 600  # default = leaf_chunk_size * 2

    # === PDF heading quantile thresholds ===
    pdf_heading_quantile_h1: float = 0.90
    pdf_heading_quantile_h2: float = 0.75
    pdf_heading_quantile_h3: float = 0.60

    # === Per-type chunking overrides (None = use global defaults) ===
    pdf_parent_max_chars: int | None = None
    pdf_leaf_chunk_size: int | None = None
    pdf_leaf_chunk_overlap: int | None = None
    md_parent_max_chars: int | None = None
    md_leaf_chunk_size: int | None = None
    md_leaf_chunk_overlap: int | None = None

    # ==================== 文件上传配置 ====================
    upload_dir: str = "data/uploads"  # 上传文件存储目录
    max_upload_size_mb: int = 50      # 最大上传文件大小（MB）

    # ==================== HuggingFace 配置 ====================
    hf_endpoint: str = "https://hf-mirror.com"  # HuggingFace 镜像地址

    # ==================== 认证配置 ====================
    jwt_secret: str = "dev-secret-change-in-production"  # JWT 签名密钥
    jwt_expire_minutes: int = 60 * 24  # Token 过期时间（默认 24 小时）

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例（带缓存）

    使用 lru_cache 装饰器缓存配置实例，避免重复读取环境变量。
    整个应用应使用此函数获取配置，而不是直接实例化 Settings。

    Returns:
        Settings 实例（单例）
    """
    return Settings()
