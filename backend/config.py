from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    deepseek_model: str = "qwen3-max"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    chroma_persist_dir: str = "./data/chroma"

    upload_dir: str = "./data/uploads"
    chunk_size: int = 500
    chunk_overlap: int = 50
    supported_extensions: list[str] = [".pdf", ".docx", ".md", ".txt"]
    max_upload_size_mb: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


settings = Settings()
