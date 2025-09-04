from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
from app.core.config import settings
from app.core.database import init_database, close_database
from app.api import chat, auth

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    try:
        await init_database()
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

app.include_router(chat.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])

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

@app.get("/health")
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
