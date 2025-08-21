from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    
    aws_region: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    embeddings_provider: str
    embeddings_model_id: str
    embeddings_region: str
    embeddings_dimensions: int
    
    llm_provider: str
    llm_model_id: str
    llm_region: str
    llm_max_tokens: int
    llm_temperature: float
    llm_top_p: float
    
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str
    
    rag_max_context_length: int
    rag_top_k_results: int
    rag_similarity_threshold: float
    
    log_level: str
    log_format: str
    
    rate_limit_requests_per_minute: int
    
    environment: str
    
    @property
    def vectordb_url(self) -> Optional[str]:
        return self.weaviate_url
    
    @property
    def vectordb_api_key(self) -> Optional[str]:
        return self.weaviate_api_key
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
