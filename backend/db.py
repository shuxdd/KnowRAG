"""
数据库连接和会话管理模块

本模块初始化 SQLAlchemy 引擎和会话工厂，连接到 PostgreSQL 数据库。
使用连接池管理数据库连接，支持高并发访问。
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.postgres_url,
    pool_size=settings.pg_pool_size,
    max_overflow=settings.pg_pool_max_overflow,
    pool_pre_ping=True,
)
SessionFactory = sessionmaker(engine, expire_on_commit=False)
