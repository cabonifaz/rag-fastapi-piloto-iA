from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from app.core.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.api_debug,  # Log SQL queries in debug mode
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=False,  # Don't test immediately during startup
    pool_recycle=3600,   # Recycle connections every hour
    connect_args={
        "timeout": 30,        # 30 second connection timeout
        "login_timeout": 30    # 30 second SQL Server login timeout
    }
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()


def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def init_database():
    """Initialize database connection and create tables if they don't exist"""
    try:
        # Test database connection with timeout
        await asyncio.wait_for(
            asyncio.to_thread(lambda: engine.connect().close()),
            timeout=15.0  # 15 second timeout for the entire operation
        )
        logger.info("Database connection successful")

    except asyncio.TimeoutError:
        logger.warning("Database connection timeout - application will continue but database features may not work")
        # Don't raise - allow app to start without database
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Application will continue but database features may not work")
        # Don't raise - allow app to start without database


async def close_database():
    """Close database connections"""
    try:
        engine.dispose()
    except Exception as e:
        logger.error(f"Error closing database: {e}")
        raise