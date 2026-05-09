from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qwen LLM
    qwen_api_key: str
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3-max"

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    # ChromaDB
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection: str = "knowledge_base"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Upload
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    # HuggingFace mirror
    hf_endpoint: str = "https://hf-mirror.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
