from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.api_debug,  # Log SQL queries in debug mode
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
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
        # Test database connection
        with engine.connect() as connection:
            logger.info("Database connection successful")
        
        # Create tables (if needed)
        # Base.metadata.create_all(bind=engine)
        # logger.info("Database tables initialized")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections"""
    try:
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")