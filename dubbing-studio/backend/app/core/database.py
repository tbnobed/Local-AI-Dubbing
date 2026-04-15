from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, inspect
from app.config import settings
from app.models.job import Base
import logging

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    async with engine.begin() as conn:
        def _migrate(sync_conn):
            inspector = inspect(sync_conn)
            existing_tables = inspector.get_table_names()

            if "jobs" not in existing_tables:
                Base.metadata.create_all(sync_conn)
                logger.info("Created database tables from scratch")
                return

            existing_cols = {c["name"] for c in inspector.get_columns("jobs")}
            model_cols = {c.name: c for c in Base.metadata.tables["jobs"].columns}

            for col_name, col_obj in model_cols.items():
                if col_name not in existing_cols:
                    col_type = col_obj.type.compile(dialect=sync_conn.dialect)
                    sync_conn.execute(
                        text(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")
                    )
                    logger.info(f"Added missing column: jobs.{col_name} ({col_type})")

        await conn.run_sync(_migrate)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
