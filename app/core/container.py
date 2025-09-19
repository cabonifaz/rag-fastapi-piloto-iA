# app/core/container.py

from app.core.config import settings
from app.services.chat_service import ChatService
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort

# Infrastructure imports
from app.infrastructure.embeddings.aws_embeddings import AWSBedrockEmbeddingsProvider
from app.infrastructure.vectorstores.weaviate_repository import WeaviateRepository
from app.infrastructure.llm.aws_provider import AWSLLMProvider


class DIContainer:
    """
    Dependency Injection Container following hexagonal architecture principles.
    Manages creation and lifecycle of all dependencies.
    """
    
    def __init__(self):
        self._embeddings_provider = None
        self._vectorstore = None
        self._llm_provider = None

    def get_embeddings_provider(self) -> EmbeddingsPort:
        """Get embeddings provider instance (singleton)."""
        if self._embeddings_provider is None:
            try:
                if settings.embeddings_provider == "aws":
                    if not settings.embeddings_region:
                        raise ValueError("Embeddings region is required for AWS provider")
                    if not settings.embeddings_model_id:
                        raise ValueError("Embeddings model ID is required for AWS provider")
                    
                    self._embeddings_provider = AWSBedrockEmbeddingsProvider(
                        region=settings.embeddings_region,
                        model_id=settings.embeddings_model_id,
                        profile_name=settings.aws_profile
                    )
                else:
                    raise ValueError(f"Unsupported embeddings provider: {settings.embeddings_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize embeddings provider: {str(e)}")
        
        return self._embeddings_provider

    def get_vectorstore(self) -> VectorStorePort:
        """Get vectorstore instance (singleton)."""
        if self._vectorstore is None:
            try:
                if not settings.vectordb_url:
                    raise ValueError("Vector database URL is required")
                if not settings.vectordb_api_key:
                    raise ValueError("Vector database API key is required")
                
                self._vectorstore = WeaviateRepository(
                    url=settings.vectordb_url,
                    api_key=settings.vectordb_api_key,
                    skip_init_checks=True
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize vector store: {str(e)}")
        return self._vectorstore

    def get_llm_provider(self) -> LLMPort:
        """Get LLM provider instance (singleton)."""
        if self._llm_provider is None:
            try:
                if settings.llm_provider == "aws":
                    if not settings.llm_region:
                        raise ValueError("LLM region is required for AWS provider")
                    if not settings.llm_model_id:
                        raise ValueError("LLM model ID is required for AWS provider")
                    
                    self._llm_provider = AWSLLMProvider(
                        region=settings.llm_region,
                        model_id=settings.llm_model_id,
                        profile_name=settings.aws_profile
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize LLM provider: {str(e)}")
        
        return self._llm_provider

    def get_full_rag_chat_service(self) -> tuple[ChatService, LLMPort]:
        """Get chat service with vectorstore AND LLM provider for complete RAG with answer generation."""
        embeddings_provider = self.get_embeddings_provider()
        vectorstore = self.get_vectorstore()
        llm_provider = self.get_llm_provider()
        
        chat_service = ChatService(
            embeddings_provider=embeddings_provider,
            vectorstore=vectorstore,
            llm_provider=llm_provider
        )
        
        return chat_service, llm_provider


# Global container instance
container = DIContainer()