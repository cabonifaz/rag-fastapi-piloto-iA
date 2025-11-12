from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from app.core.config import settings
import logging
import asyncio
from functools import wraps
import time

logger = logging.getLogger(__name__)

# Create database engine with resilient connection pooling
engine = create_engine(
    settings.database_url,
    echo=settings.api_debug,  # Log SQL queries in debug mode
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # ✅ Test connections before using them
    pool_recycle=1800,   # ✅ Recycle connections every 30 minutes (shorter interval)
    pool_reset_on_return='rollback',  # ✅ Rollback on every return for clean state
    connect_args={
        "timeout": 30,        # 30 second connection timeout
        "login_timeout": 30   # 30 second SQL Server login timeout
    }
)

# ✅ Add event listeners for connection pool health checking
@event.listens_for(pool.Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log successful connection"""
    logger.debug(f"New database connection established")

@event.listens_for(pool.Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Monitor connection checkout from pool"""
    logger.debug("Connection checked out from pool")

@event.listens_for(pool.Pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Monitor connection return to pool"""
    logger.debug("Connection returned to pool")

@event.listens_for(pool.Pool, "detach")
def receive_detach(dbapi_conn, connection_record):
    """Log when connection is detached from pool (due to errors)"""
    logger.warning(f"Connection detached from pool (likely due to error)")

@event.listens_for(pool.Pool, "close")
def receive_close(dbapi_conn, connection_record):
    """Log when connection is closed"""
    logger.debug(f"Connection closed")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()


def retry_on_db_error(max_retries=3, delay=1):
    """
    Decorator to retry database operations on connection errors.

    Args:
        max_retries: Number of retries (default: 3)
        delay: Delay between retries in seconds (default: 1)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_code = str(e)
                    # Check if it's a communication/connection error
                    if any(code in error_code for code in ['08S01', '0x274C', 'Communication link failure', 'connection']):
                        logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                            continue
                    raise
            if last_error:
                raise last_error

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_code = str(e)
                    # Check if it's a communication/connection error
                    if any(code in error_code for code in ['08S01', '0x274C', 'Communication link failure', 'connection']):
                        logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(delay * (2 ** attempt))  # Exponential backoff
                            continue
                    raise
            if last_error:
                raise last_error

        # Return async or sync wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_db() -> Session:
    """Dependency to get database session with better error handling"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Error rolling back transaction: {rollback_error}")
        raise
    finally:
        try:
            db.close()
        except Exception as close_error:
            # If it's a connection error, dispose of the connection to prevent reuse
            error_code = str(close_error)
            if any(code in error_code for code in ['08S01', '0x274C', 'Communication link failure', 'connection']):
                logger.warning(f"Connection error while closing session, disposing pool: {close_error}")
                engine.dispose()
            else:
                logger.error(f"Error closing database session: {close_error}")


async def init_database():
    """Initialize database connection and create tables if they don't exist"""
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            # Test database connection with timeout
            await asyncio.wait_for(
                asyncio.to_thread(lambda: engine.connect().close()),
                timeout=15.0  # 15 second timeout for the entire operation
            )
            logger.info("Database connection successful")
            return

        except asyncio.TimeoutError:
            logger.warning(f"Database connection timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                logger.warning("Database connection timeout after all retries - application will continue but database features may not work")

        except Exception as e:
            logger.error(f"Database initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                logger.warning("Database initialization failed after all retries - application will continue but database features may not work")


async def close_database():
    """Close database connections gracefully"""
    try:
        logger.info("Closing database connections...")
        engine.dispose()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database: {e}")