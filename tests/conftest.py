import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.db_models import Base


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:17") as pc:
        yield pc


@pytest.fixture
def pg_session(postgres_container):
    engine = create_engine(postgres_container.get_connection_url())
    Base.metadata.create_all(engine)
    yield sessionmaker(engine)
    Base.metadata.drop_all(engine)
