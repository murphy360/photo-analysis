from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False)


async def init_db() -> None:
    # Real schema migrations run via `alembic upgrade head` at container
    # startup (see the Dockerfile CMD) — that's what handles an *existing*
    # database gaining a new column. create_all here only ever creates
    # tables that don't exist yet, so it's a no-op against a database that
    # already has the table but is missing a newer column; it exists mainly
    # so the test suite (which never goes through the Docker CMD) gets a
    # schema for its fresh, empty per-run SQLite file. Whenever a model
    # field changes, generate a migration for it (`alembic revision
    # --autogenerate -m "..."` — and add `import sqlmodel` to the result,
    # autogenerate doesn't add it) or existing databases silently keep the
    # old schema and every insert starts failing.
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
        yield session
