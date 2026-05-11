import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import documents, qa, eval

# 创建 FastAPI 应用实例
app = FastAPI(
    title="KnowRAG - Enterprise Knowledge Base",  # API 标题
    version="1.0.0",                              # API 版本
    description="RAG-powered enterprise knowledge base with hybrid search and reranking",  # API 描述
)

# 配置 CORS 中间件，允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 允许的前端地址
    allow_credentials=True,                    # 允许携带凭证
    allow_methods=["*"],                       # 允许所有 HTTP 方法
    allow_headers=["*"],                       # 允许所有请求头
)

# 注册各模块的路由
app.include_router(documents.router)  # 文档管理路由
app.include_router(qa.router)         # 问答路由
app.include_router(eval.router)        # 评估路由


@app.get("/api/health")
async def health_check():
    """
    健康检查接口

    Returns:
        包含服务状态和版本信息的字典
    """
    return {"status": "ok", "version": "1.0.0"}
