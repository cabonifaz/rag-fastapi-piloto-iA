from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    
    aws_region: str
    aws_profile: Optional[str] = None
    
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
    weaviate_grpc: Optional[str] = None
    
    rag_max_context_length: int
    rag_top_k_results: int
    rag_similarity_threshold: float
    
    # SQL Server Database Configuration
    db_server: str
    db_database: str
    db_username: str
    db_password: str
    db_driver: str = "ODBC Driver 17 for SQL Server"
    db_port: int = 1433
    db_trusted_connection: bool = False
    
    log_level: str
    log_format: str
    
    rate_limit_requests_per_minute: int
    
    environment: str
    
    @validator('aws_region')
    def validate_aws_region(cls, v):
        if not v:
            raise ValueError("AWS region is required")
        return v
    
    @validator('embeddings_provider')
    def validate_embeddings_provider(cls, v):
        if v not in ['aws', 'openai']:
            raise ValueError("Embeddings provider must be 'aws' or 'openai'")
        return v
    
    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        if v not in ['aws', 'openai']:
            raise ValueError("LLM provider must be 'aws' or 'openai'")
        return v
    
    @validator('embeddings_model_id')
    def validate_embeddings_model_id(cls, v):
        if not v:
            raise ValueError("Embeddings model ID is required")
        return v
    
    @validator('llm_model_id')
    def validate_llm_model_id(cls, v):
        if not v:
            raise ValueError("LLM model ID is required")
        return v
    
    @validator('llm_max_tokens')
    def validate_llm_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("LLM max tokens must be greater than 0")
        return v
    
    @validator('llm_temperature')
    def validate_llm_temperature(cls, v):
        if not (0.0 <= v <= 2.0):
            raise ValueError("LLM temperature must be between 0.0 and 2.0")
        return v
    
    @validator('rag_top_k_results')
    def validate_rag_top_k_results(cls, v):
        if v <= 0:
            raise ValueError("RAG top_k results must be greater than 0")
        return v
    
    @validator('rag_similarity_threshold')
    def validate_rag_similarity_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("RAG similarity threshold must be between 0.0 and 1.0")
        return v
    
    @validator('db_server')
    def validate_db_server(cls, v):
        if not v:
            raise ValueError("Database server is required")
        return v
    
    @validator('db_database')
    def validate_db_database(cls, v):
        if not v:
            raise ValueError("Database name is required")
        return v
    
    @validator('db_username')
    def validate_db_username(cls, v):
        if not v:
            raise ValueError("Database username is required")
        return v
    
    @validator('db_password')
    def validate_db_password(cls, v):
        if not v:
            raise ValueError("Database password is required")
        return v
    
    @property
    def vectordb_url(self) -> Optional[str]:
        return self.weaviate_url
    
    @property
    def vectordb_api_key(self) -> Optional[str]:
        return self.weaviate_api_key
    
    @property
    def database_url(self) -> str:
        """Generate SQL Server connection string"""
        if self.db_trusted_connection:
            return f"mssql+pyodbc://@{self.db_server}:{self.db_port}/{self.db_database}?driver={self.db_driver.replace(' ', '+')}&trusted_connection=yes"
        else:
            return f"mssql+pyodbc://{self.db_username}:{self.db_password}@{self.db_server}:{self.db_port}/{self.db_database}?driver={self.db_driver.replace(' ', '+')}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise RuntimeError(f"Configuration validation failed: {str(e)}")
