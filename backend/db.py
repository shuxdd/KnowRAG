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
