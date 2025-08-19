from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    
    # AWS Configuration
    aws_region: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Embeddings Configuration
    embeddings_provider: str
    embeddings_model_id: str
    embeddings_region: str
    embeddings_dimensions: int
    
    # LLM Configuration
    llm_provider: str
    llm_model_id: str
    llm_region: str
    llm_max_tokens: int
    llm_temperature: float
    llm_top_p: float
    
    # Weaviate Configuration
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str
    
    # RAG Configuration
    rag_max_context_length: int
    rag_top_k_results: int
    rag_similarity_threshold: float
    
    # Logging
    log_level: str
    log_format: str
    
    # Rate Limiting
    rate_limit_requests_per_minute: int
    
    # Environment
    environment: str
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
