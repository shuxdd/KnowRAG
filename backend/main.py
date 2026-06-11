"""
KnowRAG 后端主入口

FastAPI 应用的主模块，负责：
1. 创建 FastAPI 应用实例
2. 配置 CORS 中间件
3. 注册各模块路由
4. 执行启动任务（数据库迁移、模型预加载）
5. 提供健康检查接口

路由分组：
- /api/auth: 认证路由（公开）
- /api/documents: 文档管理路由（需认证）
- /api/qa: 问答路由（需认证）
"""

import os
import warnings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Suppress noisy deprecation warnings from third-party libraries
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus.*")

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import documents, qa, auth_router, knowledge_graph
from backend.utils.auth import get_current_user
from backend.services.reranker import reranker  # noqa: F401 — eager-load at import time

app = FastAPI(
    title="KnowRAG - Enterprise Knowledge Base",
    version="1.0.0",
    description="RAG-powered enterprise knowledge base with hybrid search and reranking",
)

import logging
from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command
from backend.config import get_settings
from backend.utils.logging import setup_logging

settings = get_settings()
setup_logging(level=settings.log_level)

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def run_startup():
    """
    启动时执行的任务

    包括：
    1. Alembic 数据库迁移
    2. 模型预加载（Reranker、Embedding 模型）
    """
    if settings.auto_migrate:
        try:
            alembic_cfg = AlembicConfig("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", settings.postgres_url)
            alembic_command.upgrade(alembic_cfg, "head")
            logger.info("Alembic migration up to date")
        except Exception as e:
            logger.warning(f"Alembic migration failed (PG not ready?): {e}")

    # 2. Preload models
    if settings.preload_models:
        logger.info("Preloading models...")
        try:
            from backend.services.reranker import reranker
            _ = reranker.model  # trigger CrossEncoder load
            logger.info("Reranker model loaded")
        except Exception as e:
            logger.warning(f"Reranker preload failed: {e}")

        try:
            from backend.services.vector_service import vector_service
            _ = vector_service.similarity_search("warmup", k=1)
            logger.info("Embedding warm-up complete")
        except Exception as e:
            logger.warning(f"Embedding warm-up failed: {e}")

        logger.info("Model preloading complete")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(documents.router, dependencies=[Depends(get_current_user)])
app.include_router(qa.router, dependencies=[Depends(get_current_user)])
app.include_router(knowledge_graph.router, dependencies=[Depends(get_current_user)])


@app.get("/api/health")
async def health_check():
    """
    健康检查接口

    Returns:
        包含服务状态和版本信息的字典
    """
    return {"status": "ok", "version": "1.0.0"}
