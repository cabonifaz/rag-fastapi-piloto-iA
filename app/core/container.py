# app/core/container.py

from app.core.config import settings
from app.services.rag_service import RagService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.message_service import MessageService
from app.services.ia_config_service import IaConfigService
from app.services.ia_models_service import IAModelsService
from app.services.company_service import CompanyService
from app.services.area_service import AreaService
from app.services.users_service import UsersService
from app.services.agents_service import AgentsService
from app.services.phone_code_service import PhoneCodeService
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.llm_nonstreaming_port import LLMNonStreamingPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.domain.ports.recontextualizer_port import RecontextualizerPort
from app.domain.ports.state_builder import StateBuilderPort
from app.domain.ports.query_rewriter import QueryRewriterPort
from app.domain.ports.transcribe_port import TranscribePort
from app.domain.ports.file_transcribe_port import FileTranscribePort
from app.domain.ports.blob_storage_port import BlobStoragePort

# Infrastructure imports
from app.infrastructure.embeddings.aws_embeddings import AWSBedrockEmbeddingsProvider
from app.infrastructure.vectorstores.weaviate_repository import WeaviateRepository
from app.infrastructure.llm.aws_bedrock_converse_provider import AWSBedrockConverseProvider
from app.infrastructure.llm.aws_bedrock_converse_provider_nonstreaming import AWSBedrockConverseNonStreamingProvider
from app.infrastructure.llm.aws_bedrock_converse_provider_llm_only_nonstreaming import AWSBedrockConverseProviderLLMOnly
from app.infrastructure.task_decomposition.aws_bedrock_provider import OrchestratorQueryAnalyzer
from app.infrastructure.recontextualizer.aws_bedrock_provider import QueryRecontextualizer
from app.infrastructure.state_builder.aws_bedrock_provider import StateBuilder
from app.infrastructure.query_rewriter.aws_bedrock_provider import QueryRewriter
from app.infrastructure.transcriber.aws_transcribe_streaming import AWSTranscribeStreaming
from app.infrastructure.transcriber.openai_transcribe import OpenAITranscribe
from app.infrastructure.blob_storages.s3_storage import S3BlobStorage


class DIContainer:
    """
    Dependency Injection Container following hexagonal architecture principles.
    Manages creation and lifecycle of all dependencies.
    """
    
    def __init__(self):
        self._embeddings_provider = None
        self._vectorstore = None
        self._llm_provider = None
        self._llm_nonstreaming_provider = None
        self._llm_only_provider = None
        self._orchestrator_analyzer = None
        self._blob_storage = None
        self._rag_service = None
        self._auth_service = None
        self._chat_service = None
        self._message_service = None
        self._recontextualizer = None
        self._state_builder = None
        self._query_rewriter = None
        self._ia_config_service = None
        self._ia_models_service = None
        self._company_service = None
        self._area_service = None
        self._users_service = None
        self._agents_service = None
        self._phone_code_service = None

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

    def get_blob_storage(self) -> BlobStoragePort:
        """Get blob storage instance (singleton)."""
        if self._blob_storage is None:
            try:
                if not settings.aws_region:
                    raise ValueError("AWS region is required for S3 blob storage")

                self._blob_storage = S3BlobStorage(
                    region=settings.aws_region,
                    profile_name=settings.aws_profile,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize blob storage: {str(e)}")
        return self._blob_storage

    def get_llm_provider(self) -> LLMPort:
        """Get LLM provider instance (singleton)."""
        if self._llm_provider is None:
            try:
                if settings.llm_provider == "aws":
                    if not settings.llm_region:
                        raise ValueError("LLM region is required for AWS provider")

                    # Using Converse API - stateless (no message history)
                    # Initialize with default model, actual model_id comes per-request from database
                    self._llm_provider = AWSBedrockConverseProvider(
                        region=settings.llm_region,
                        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Default/fallback model
                        role_behavior=None,  # Will be provided per-request from database
                        profile_name=settings.aws_profile,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize LLM provider: {str(e)}")

        return self._llm_provider

    def get_llm_nonstreaming_provider(self) -> LLMNonStreamingPort:
        """Get non-streaming LLM provider instance for RAG mode (singleton)."""
        if self._llm_nonstreaming_provider is None:
            try:
                if settings.llm_provider == "aws":
                    if not settings.llm_region:
                        raise ValueError("LLM region is required for AWS provider")

                    # Using non-streaming Converse API for n8n RAG mode
                    # Initialize with default model, actual model_id comes per-request from database
                    self._llm_nonstreaming_provider = AWSBedrockConverseNonStreamingProvider(
                        region=settings.llm_region,
                        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Default/fallback model
                        role_behavior=None,  # Will be provided per-request from database
                        profile_name=settings.aws_profile,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize non-streaming LLM provider: {str(e)}")

        return self._llm_nonstreaming_provider

    def get_llm_only_provider(self) -> LLMNonStreamingPort:
        """Get LLM-only provider instance for conversational mode without RAG (singleton)."""
        if self._llm_only_provider is None:
            try:
                if settings.llm_provider == "aws":
                    if not settings.llm_region:
                        raise ValueError("LLM region is required for AWS provider")

                    # Using non-streaming Converse API for LLM-only mode (no RAG)
                    # Initialize with default model, actual model_id comes per-request from database
                    self._llm_only_provider = AWSBedrockConverseProviderLLMOnly(
                        region=settings.llm_region,
                        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Default/fallback model
                        role_behavior=None,  # Will be provided per-request from database
                        profile_name=settings.aws_profile,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize LLM-only provider: {str(e)}")

        return self._llm_only_provider

    def get_rag_service(self) -> RagService:
        """Get rag service as singleton (stateless, no db parameter)."""
        if self._rag_service is None:
            embeddings_provider = self.get_embeddings_provider()
            vectorstore = self.get_vectorstore()
            llm_provider = self.get_llm_provider()
            llm_nonstreaming_provider = self.get_llm_nonstreaming_provider()
            llm_only_provider = self.get_llm_only_provider()
            message_service = self.get_message_service()
            ia_config_service = self.get_ia_config_service()
            state_builder = self.get_state_builder()
            query_rewriter = self.get_query_rewriter()
            orchestrator = self.get_orchestrator_analyzer()

            # Create ONCE - singleton with all dependencies injected
            self._rag_service = RagService(
                embeddings_provider=embeddings_provider,
                vectorstore=vectorstore,
                llm_provider=llm_provider,
                message_service=message_service,
                ia_config_service=ia_config_service,
                state_builder=state_builder,
                query_rewriter=query_rewriter,
                orchestrator=orchestrator,
                llm_nonstreaming_provider=llm_nonstreaming_provider,
                llm_only_provider=llm_only_provider
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

    def get_state_builder(self) -> StateBuilderPort:
        """Get state builder as singleton."""
        if self._state_builder is None:
            # Create ONCE - singleton
            # Uses settings for AWS configuration
            self._state_builder = StateBuilder()

        return self._state_builder

    def get_query_rewriter(self) -> QueryRewriterPort:
        """Get query rewriter as singleton."""
        if self._query_rewriter is None:
            # Create ONCE - singleton
            # Uses settings for AWS configuration
            self._query_rewriter = QueryRewriter()

        return self._query_rewriter

    def get_ia_config_service(self) -> IaConfigService:
        """Get IA config service as singleton (stateless, no db parameter)."""
        if self._ia_config_service is None:
            # Create ONCE - singleton
            self._ia_config_service = IaConfigService()

        return self._ia_config_service

    def get_ia_models_service(self) -> IAModelsService:
        """Get IA models service as singleton (stateless, no db parameter)."""
        if self._ia_models_service is None:
            # Create ONCE - singleton
            self._ia_models_service = IAModelsService()

        return self._ia_models_service

    def get_company_service(self) -> CompanyService:
        """Get company service as singleton (stateless, no db parameter)."""
        if self._company_service is None:
            # Create ONCE - singleton
            self._company_service = CompanyService()

        return self._company_service

    def get_area_service(self) -> AreaService:
        """Get area service as singleton (stateless, no db parameter)."""
        if self._area_service is None:
            # Create ONCE - singleton
            self._area_service = AreaService()

        return self._area_service

    def get_users_service(self) -> UsersService:
        """Get users service as singleton (stateless, no db parameter)."""
        if self._users_service is None:
            # Create ONCE - singleton
            self._users_service = UsersService()

        return self._users_service

    def get_agents_service(self) -> AgentsService:
        """Get agents service as singleton (stateless, no db parameter)."""
        if self._agents_service is None:
            # Create ONCE - singleton
            self._agents_service = AgentsService()

        return self._agents_service

    def get_phone_code_service(self) -> PhoneCodeService:
        """Get phone code service as singleton (stateless, no db parameter)."""
        if self._phone_code_service is None:
            # Create ONCE - singleton
            self._phone_code_service = PhoneCodeService()

        return self._phone_code_service

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