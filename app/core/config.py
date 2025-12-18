from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict
from typing import Optional
import logging
from .database_config import database_config

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    
    aws_region: str
    aws_profile: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    embeddings_provider: str
    embeddings_model_id: str
    embeddings_region: str
    embeddings_dimensions: int
    
    llm_provider: str
    llm_region: str
    llm_top_p: float

    orchestrator_model_id: str
    orchestrator_max_tokens: int
    orchestrator_temperature: float
    orchestrator_top_p: float

    recontextualizer_model_id: str

    # Transcribe Configuration
    transcribe_language_code: str
    transcribe_sample_rate: int
    transcribe_media_encoding: str

    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_transcribe_model: Optional[str] = None
    openai_transcribe_timeout: Optional[int] = None

    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str
    weaviate_grpc: Optional[str] = None

    log_level: str

    environment: str
    
    # JWT Configuration
    jwt_secret_key: str
    jwt_expiration_minutes: int = 2880
    
    # CORS Configuration
    cors_origins: str

    # DynamoDB Configuration
    dynamodb_table_messages: str

    # S3 Configuration
    s3_pdfs_bucket: str
    s3_ingest_results_bucket: str

    # n8n Webhook Configuration
    n8n_cc_webhook_url: Optional[str] = None
    n8n_cc_jwt_secret: Optional[str] = None


    s3_logos_bucket: str
    
    
    @field_validator('aws_region')
    @classmethod
    def validate_aws_region(cls, v):
        if not v:
            raise ValueError("AWS region is required")
        return v
    
    @field_validator('embeddings_provider')
    @classmethod
    def validate_embeddings_provider(cls, v):
        if v not in ['aws', 'openai']:
            raise ValueError("Embeddings provider must be 'aws' or 'openai'")
        return v
    
    @field_validator('llm_provider')
    @classmethod
    def validate_llm_provider(cls, v):
        if v not in ['aws', 'openai']:
            raise ValueError("LLM provider must be 'aws' or 'openai'")
        return v
    
    @field_validator('embeddings_model_id')
    @classmethod
    def validate_embeddings_model_id(cls, v):
        if not v:
            raise ValueError("Embeddings model ID is required")
        return v
    
    @field_validator('recontextualizer_model_id')
    @classmethod
    def validate_recontextualizer_model_id(cls, v):
        if not v:
            raise ValueError("Recontextualizer model ID is required")
        return v

    @field_validator('jwt_secret_key')
    @classmethod
    def validate_jwt_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    @field_validator('jwt_expiration_minutes')
    @classmethod
    def validate_jwt_expiration_minutes(cls, v):
        if v <= 0 or v > 2880:
            raise ValueError("JWT expiration minutes must be between 1 and 1440")
        return v

    @field_validator('transcribe_sample_rate')
    @classmethod
    def validate_transcribe_sample_rate(cls, v):
        valid_rates = [8000, 16000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Transcribe sample rate must be one of {valid_rates}")
        return v

    @field_validator('transcribe_media_encoding')
    @classmethod
    def validate_transcribe_media_encoding(cls, v):
        valid_encodings = ['pcm', 'ogg-opus', 'flac']
        if v not in valid_encodings:
            raise ValueError(f"Transcribe media encoding must be one of {valid_encodings}")
        return v

    @field_validator('transcribe_language_code')
    @classmethod
    def validate_transcribe_language_code(cls, v):
        if not v or len(v) < 5 or '-' not in v:
            raise ValueError("Transcribe language code must be in format 'xx-XX' (e.g., 'es-ES', 'en-US')")
        return v

    @property
    def vectordb_url(self) -> Optional[str]:
        return self.weaviate_url
    
    @property
    def vectordb_api_key(self) -> Optional[str]:
        return self.weaviate_api_key
    
    @property
    def database_url(self) -> str:
        """Get database connection string from database config"""
        return database_config.database_url

try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise RuntimeError(f"Configuration validation failed: {str(e)}")
