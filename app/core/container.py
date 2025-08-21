# app/core/container.py

from app.core.config import settings
from app.application.chat_service import ChatService
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
        self._chat_service = None

    def get_embeddings_provider(self) -> EmbeddingsPort:
        """Get embeddings provider instance (singleton)."""
        if self._embeddings_provider is None:
            if settings.embeddings_provider == "aws":
                self._embeddings_provider = AWSBedrockEmbeddingsProvider(
                    region=settings.embeddings_region,
                    model_id=settings.embeddings_model_id,
                    access_key=settings.aws_access_key_id,
                    secret_key=settings.aws_secret_access_key
                )
            else:
                raise ValueError(f"Unsupported embeddings provider: {settings.embeddings_provider}")
        
        return self._embeddings_provider

    def get_vectorstore(self) -> VectorStorePort:
        """Get vectorstore instance (singleton)."""
        if self._vectorstore is None:
            self._vectorstore = WeaviateRepository(
                url=settings.vectordb_url,
                api_key=settings.vectordb_api_key
            )
        return self._vectorstore

    def get_llm_provider(self) -> LLMPort:
        """Get LLM provider instance (singleton)."""
        if self._llm_provider is None:
            if settings.llm_provider == "aws":
                self._llm_provider = AWSLLMProvider(
                    region=settings.llm_region,
                    model_id=settings.llm_model_id,
                    access_key=settings.aws_access_key_id,
                    secret_key=settings.aws_secret_access_key
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
        
        return self._llm_provider

    def get_chat_service(self) -> ChatService:
        """Get chat service instance (singleton)."""
        if self._chat_service is None:
            embeddings_provider = self.get_embeddings_provider()
            # Don't initialize vectorstore unless needed to avoid connection errors
            vectorstore = None
            
            self._chat_service = ChatService(
                embeddings_provider=embeddings_provider,
                vectorstore=vectorstore
            )
        
        return self._chat_service

    def get_chat_service_with_vectorstore(self) -> ChatService:
        """Get chat service with vectorstore for full RAG functionality."""
        embeddings_provider = self.get_embeddings_provider()
        vectorstore = self.get_vectorstore()
        
        return ChatService(
            embeddings_provider=embeddings_provider,
            vectorstore=vectorstore
        )

    def get_full_rag_chat_service(self) -> tuple[ChatService, LLMPort]:
        """Get chat service with vectorstore AND LLM provider for complete RAG with answer generation."""
        embeddings_provider = self.get_embeddings_provider()
        vectorstore = self.get_vectorstore()
        llm_provider = self.get_llm_provider()
        
        chat_service = ChatService(
            embeddings_provider=embeddings_provider,
            vectorstore=vectorstore
        )
        
        return chat_service, llm_provider


# Global container instance
container = DIContainer()