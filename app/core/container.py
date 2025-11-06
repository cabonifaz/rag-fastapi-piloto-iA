# app/core/container.py

from app.core.config import settings
from app.services.rag_service import RagService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.message_service import MessageService
from app.services.ia_config_service import IaConfigService
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.domain.ports.recontextualizer_port import RecontextualizerPort
from app.domain.ports.transcribe_port import TranscribePort
from app.domain.ports.file_transcribe_port import FileTranscribePort

# Infrastructure imports
from app.infrastructure.embeddings.aws_embeddings import AWSBedrockEmbeddingsProvider
from app.infrastructure.vectorstores.weaviate_repository import WeaviateRepository
from app.infrastructure.llm.aws_bedrock_converse_provider import AWSBedrockConverseProvider
from app.infrastructure.task_decomposition.aws_bedrock_provider import OrchestratorQueryAnalyzer
from app.infrastructure.recontextualizer.aws_bedrock_provider import QueryRecontextualizer
from app.infrastructure.transcriber.aws_transcribe_streaming import AWSTranscribeStreaming
from app.infrastructure.transcriber.openai_transcribe import OpenAITranscribe


class DIContainer:
    """
    Dependency Injection Container following hexagonal architecture principles.
    Manages creation and lifecycle of all dependencies.
    """
    
    def __init__(self):
        self._embeddings_provider = None
        self._vectorstore = None
        self._llm_provider = None
        self._orchestrator_analyzer = None
        self._rag_service = None
        self._auth_service = None
        self._chat_service = None
        self._message_service = None
        self._recontextualizer = None
        self._ia_config_service = None

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
                        profile_name=settings.aws_profile,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key
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

                    # Using Converse API - stateless (no message history)
                    # Each request is independent with only current user prompt
                    self._llm_provider = AWSBedrockConverseProvider(
                        region=settings.llm_region,
                        model_id=settings.llm_model_id,
                        role_behavior=settings.llm_role_behavior,
                        profile_name=settings.aws_profile,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize LLM provider: {str(e)}")

        return self._llm_provider

    def get_rag_service(self) -> RagService:
        """Get rag service as singleton (stateless, no db parameter)."""
        if self._rag_service is None:
            embeddings_provider = self.get_embeddings_provider()
            vectorstore = self.get_vectorstore()
            llm_provider = self.get_llm_provider()
            message_service = self.get_message_service()
            ia_config_service = self.get_ia_config_service()
            recontextualizer = self.get_recontextualizer()
            orchestrator = self.get_orchestrator_analyzer()

            # Create ONCE - singleton with all dependencies injected
            self._rag_service = RagService(
                embeddings_provider=embeddings_provider,
                vectorstore=vectorstore,
                llm_provider=llm_provider,
                message_service=message_service,
                ia_config_service=ia_config_service,
                recontextualizer=recontextualizer,
                orchestrator=orchestrator
            )

        return self._rag_service

    def get_auth_service(self) -> AuthService:
        """Get auth service as singleton (stateless, no db parameter)."""
        if self._auth_service is None:
            # Create ONCE - singleton
            self._auth_service = AuthService()

        return self._auth_service

    def get_orchestrator_analyzer(self) -> OrchestratorQueryAnalyzer:
        """Get orchestrator query analyzer instance (singleton)."""
        if self._orchestrator_analyzer is None:
            try:
                self._orchestrator_analyzer = OrchestratorQueryAnalyzer()
            except Exception as e:
                raise ConnectionError(f"Failed to initialize orchestrator analyzer: {str(e)}")

        return self._orchestrator_analyzer

    def get_chat_service(self) -> ChatService:
        """Get chat service as singleton (stateless, no db parameter)."""
        if self._chat_service is None:
            # Create ONCE - singleton
            self._chat_service = ChatService()

        return self._chat_service

    def get_message_service(self) -> MessageService:
        """Get message service as singleton (stateless, uses DynamoDB)."""
        if self._message_service is None:
            # Create ONCE - singleton
            self._message_service = MessageService()

        return self._message_service

    def get_recontextualizer(self) -> RecontextualizerPort:
        """Get query recontextualizer as singleton."""
        if self._recontextualizer is None:
            # Create ONCE - singleton
            # Uses settings for AWS configuration
            self._recontextualizer = QueryRecontextualizer()

        return self._recontextualizer

    def get_ia_config_service(self) -> IaConfigService:
        """Get IA config service as singleton (stateless, no db parameter)."""
        if self._ia_config_service is None:
            # Create ONCE - singleton
            self._ia_config_service = IaConfigService()

        return self._ia_config_service

    def create_transcribe_session(self) -> TranscribePort:
        """
        Create NEW transcribe session for a single user/WebSocket connection.

        🚨 CRITICAL: This is a FACTORY method, NOT a singleton getter.
            → Returns a NEW instance every time it's called
            → Each WebSocket connection must call this to get its own instance
            → DO NOT cache or reuse instances across connections

        Uses AWS Transcribe Streaming for real-time WebSocket transcription.

        Returns:
            TranscribePort: NEW AWS Transcribe streaming instance
        """
        try:
            if not settings.aws_region:
                raise ValueError("AWS region is required for AWS Transcribe service")

            # Create NEW AWS Transcribe instance - not a singleton!
            return AWSTranscribeStreaming(
                region=settings.aws_region,
                profile_name=settings.aws_profile,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create transcribe session: {str(e)}")

    def create_file_transcribe_session(self) -> FileTranscribePort:
        """
        Create NEW file-based transcribe session for file uploads (OpenAI).

        This is for batch file transcription (not streaming WebSocket).
        Use this for frontend file uploads.

        Returns:
            FileTranscribePort: NEW OpenAI Transcribe instance for file processing
        """
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI Transcribe service")

            # Create NEW OpenAI Transcribe instance - not a singleton!
            return OpenAITranscribe(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or "https://api.openai.com/v1",
                model=settings.openai_transcribe_model or "gpt-4o-mini-transcribe",
                timeout=settings.openai_transcribe_timeout or 60
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create file transcribe session: {str(e)}")


# Global container instance
container = DIContainer()