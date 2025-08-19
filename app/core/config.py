# app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Vector DB
    VECTORDB_URL: str = os.getenv("VECTORDB_URL", "")
    VECTORDB_API_KEY: str = os.getenv("VECTORDB_API_KEY", "")

    # Embeddings
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "")
    EMBEDDINGS_API_KEY: str = os.getenv("EMBEDDINGS_API_KEY", "")

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_REGION: str = os.getenv("LLM_REGION", "")
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "")

    # Memoria
    MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "inmemory")

    # Otros
    APP_NAME: str = os.getenv("APP_NAME", "rag-fastapi")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
