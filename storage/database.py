"""
Database connection and session management
Async SQLAlchemy setup with connection pooling
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from config.settings import settings
from storage.models import Base

logger = logging.getLogger(__name__)


# ============================================
# ENGINE CONFIGURATION
# ============================================

# Create async engine with connection pooling
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    poolclass=QueuePool,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================
# DATABASE INITIALIZATION
# ============================================

async def init_db():
    """
    Initialize database: create tables and enable pgvector extension
    """
    try:
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

            # Create additional indexes
            await _create_additional_indexes(conn)

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def _create_additional_indexes(conn):
    """Create additional performance indexes"""
    try:
        # Full-text search indexes for content
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pages_content_fts
            ON scraped_pages USING gin (to_tsvector('english', content))
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pages_title_fts
            ON scraped_pages USING gin (to_tsvector('english', title))
        """))

        logger.info("Additional indexes created successfully")
    except Exception as e:
        logger.warning(f"Error creating additional indexes: {e}")


async def drop_db():
    """Drop all tables (use with caution!)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped")


async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")


# ============================================
# SESSION MANAGEMENT
# ============================================

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI endpoints

    Usage:
        @app.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


# ============================================
# HEALTH CHECK
# ============================================

async def check_db_health() -> bool:
    """Check if database connection is healthy"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# ============================================
# MIGRATION HELPERS
# ============================================

async def get_db_version() -> str:
    """Get PostgreSQL version"""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            return version
    except Exception as e:
        logger.error(f"Error getting database version: {e}")
        return "Unknown"


async def check_pgvector_installed() -> bool:
    """Check if pgvector extension is installed"""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            ))
            return result.scalar()
    except Exception as e:
        logger.error(f"Error checking pgvector: {e}")
        return False


# ============================================
# UTILITIES
# ============================================

async def execute_raw_sql(sql: str, params: dict = None):
    """Execute raw SQL query"""
    async with get_session() as session:
        result = await session.execute(text(sql), params or {})
        return result


async def get_table_count(table_name: str) -> int:
    """Get row count for a table"""
    async with get_session() as session:
        result = await session.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        )
        return result.scalar()


async def vacuum_analyze():
    """Run VACUUM ANALYZE to optimize database"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("VACUUM ANALYZE"))
        logger.info("Database vacuumed and analyzed")
    except Exception as e:
        logger.error(f"Error running VACUUM ANALYZE: {e}")
