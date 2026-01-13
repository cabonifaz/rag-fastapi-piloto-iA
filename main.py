from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os
import logging

# Configure logging FIRST - load settings to get LOG_LEVEL
from dotenv import load_dotenv
load_dotenv()

# Get log level from environment (default to INFO if not set)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging with the specified level
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure logging to filter health checks AFTER basic config
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Check if this is an access log record for health check endpoint
        if hasattr(record, 'args') and record.args and len(record.args) >= 3:
            return record.args[2] != "/api/v1/health/check"
        return True

# Apply filter immediately
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

from app.core.config import settings
# Initialize database FIRST, before heavy imports
from app.core.database import init_database, close_database, SessionLocal

# Then import the heavy modules
from app.api import rag, auth, external_login, chats, messages, transcribe, file_transcribe, company, area, users, agents, knowledge, ia_models, ia_config, phone_code, menu_items
from app.core.container import container

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    try:
        await init_database()

        # Initialize and compile workflows once at startup
        # Workflows use SessionLocal factory to get sessions from pool per-request
        container.initialize_rag_workflow()
        container.initialize_llm_only_workflow()
        container.initialize_rag_anonymous_workflow()

        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

    yield

    # Shutdown
    try:
        await close_database()
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


app = FastAPI(
    title="Qamaq RAG API",
    description="Retrieval-Augmented Generation API with AWS Bedrock and Weaviate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# IP filtering disabled - using CORS only for web access control

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),  # From environment variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(external_login.router, prefix="/api/v1", tags=["external-login"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chat-management"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["message-management"])
app.include_router(transcribe.router, prefix="/api/v1/transcribe", tags=["transcription"])
app.include_router(file_transcribe.router, prefix="/api/v1", tags=["file-transcription"])
app.include_router(company.router, prefix="/api/v1/company", tags=["company-management"])
app.include_router(area.router, prefix="/api/v1/area", tags=["area-management"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users-management"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents-management"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge-management"])
app.include_router(ia_models.router, prefix="/api/v1/ia_models", tags=["ia-models-management"])
app.include_router(ia_config.router, prefix="/api/v1/ia_config", tags=["ia-config-management"])
app.include_router(phone_code.router, prefix="/api/v1/phone_code", tags=["phone-code-management"])
app.include_router(menu_items.router, prefix="/api/v1/menu", tags=["menu-management"])



@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request data", "details": exc.errors()}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"Value error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "details": str(exc)}
    )

@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    logger.error(f"Connection error: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={"error": "Service unavailable", "details": str(exc)}
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    logger.error(f"Timeout error: {str(exc)}")
    return JSONResponse(
        status_code=504,
        content={"error": "Request timeout", "details": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": "An unexpected error occurred"}
    )

@app.get("/")
async def root():
    return {"message": "Welcome to Qamaq RAG API", "version": "1.0.0"}

@app.get("/api/v1/health/check")
async def health_check():
    try:
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "service": "qamaq-rag-api"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "qamaq-rag-api", "error": str(e)}
        )

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload,
            log_level=settings.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise RuntimeError(f"Server startup failed: {str(e)}")
