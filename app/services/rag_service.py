from typing import Dict, Any, AsyncGenerator, Optional
import logging
import asyncio
from sqlalchemy.orm import Session

# Domain ports (for dependency injection in __init__)
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.llm_nonstreaming_port import LLMNonStreamingPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.domain.ports.recontextualizer_port import RecontextualizerPort
from app.domain.ports.context_gatekeeper_port import ContextGatekeeperPort

# Services (for dependency injection in __init__)
from app.services.message_service import MessageService
from app.services.ia_config_service import IaConfigService

# TTS streaming workflow
from app.workflows.tts_streaming import stream_with_tts

# Workflows (everything else is in here now!)
from app.workflows.rag_workflow import (
    get_compiled_rag_workflow,
    build_initial_state,
    stream_workflow_progress,
    validate_workflow_state,
    generate_metadata_events,
    stream_llm_response
)
from app.workflows.vlm_workflow import get_compiled_vlm_workflow
from app.workflows.helpers.service_helpers import build_vlm_initial_state, validate_vlm_workflow_state
from app.workflows.helpers.streaming_helpers import stream_vlm_response
from app.workflows.llm_only_workflow import (
    get_compiled_llm_only_workflow,
    build_llm_only_initial_state,
    validate_llm_only_workflow_state,
    generate_complete_llm_only_response
)
from app.workflows.rag_anonymous_workflow import (
    get_compiled_rag_anonymous_workflow,
    build_initial_state_anonymous,
    validate_workflow_state_anonymous,
    generate_complete_llm_response_anonymous
)
from app.workflows.llm_only_anonymous_workflow import (
    get_compiled_llm_only_anonymous_workflow,
    build_llm_only_initial_state_anonymous,
    validate_llm_only_workflow_state_anonymous,
    generate_complete_llm_only_response_anonymous
)
from app.workflows.helpers import execute_workflow_n8n, generate_complete_llm_response

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NO_CONTEXT_MESSAGE = "Parece que tu pregunta no es lo suficientemente específica 🤔. ¿Me das un poco más de contexto para ayudarte mejor?"


class RagService:
    """
    Application service that orchestrates RAG flow following hexagonal architecture:
    1. Generates embeddings using the embeddings port
    2. Searches context in vector database using vectorstore port
    3. Calls LLM with context
    4. Returns response
    """

    def __init__(
        self,
        embeddings_provider: EmbeddingsPort,
        vectorstore: VectorStorePort,
        llm_provider: LLMPort,
        message_service: MessageService,
        ia_config_service: IaConfigService,
        recontextualizer: RecontextualizerPort,
        context_gatekeeper: ContextGatekeeperPort,
        orchestrator: Optional[QueryAnalysisPort] = None,
        llm_nonstreaming_provider: LLMNonStreamingPort = None,
        llm_only_provider: LLMNonStreamingPort = None,
        tts_provider = None,  # Optional: inject for connection reuse
        vlm_provider = None,
    ):
        """
        Initialize RagService with all dependencies injected.

        Args:
            embeddings_provider: Port for generating embeddings
            vectorstore: Port for vector database operations
            llm_provider: Port for LLM operations (streaming)
            message_service: Service for managing chat messages in DynamoDB
            ia_config_service: Service for loading IA area configuration
            orchestrator: Optional port for query analysis and task decomposition
            llm_nonstreaming_provider: Optional port for non-streaming LLM operations (for n8n RAG mode)
            llm_only_provider: Optional port for non-streaming LLM operations in LLM-only mode (no RAG)
            tts_provider: Optional TTS provider for connection reuse (recommended for production)
        """
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
        self.llm_provider = llm_provider
        self.message_service = message_service
        self.ia_config_service = ia_config_service
        self.recontextualizer = recontextualizer
        self.context_gatekeeper = context_gatekeeper
        self.orchestrator = orchestrator
        self.llm_nonstreaming_provider = llm_nonstreaming_provider
        self.llm_only_provider = llm_only_provider
        self.tts_provider = tts_provider
        self.vlm_provider = vlm_provider


    async def process_rag_query_stream(
        self,
        user_id: int,
        message: str,
        company_id: int,
        area_id: int,
        db: Session,
        created_at: str,
        chat_id: str = None,
        request_timezone: str = None,
        tts: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Proceso RAG completo con streaming: embeddings → search → LLM streaming → response
        Uses pre-compiled LangGraph workflow for modular processing with progress updates.
        """
        try:
            # Get pre-compiled workflow
            app = get_compiled_rag_workflow()

            # Build initial state
            initial_state = build_initial_state(
                user_id=user_id,
                message=message,
                company_id=company_id,
                area_id=area_id,
                created_at=created_at,
                chat_id=chat_id,
                request_timezone=request_timezone
            )

            # Stream workflow progress
            result_state = None
            async for event_type, event_data in stream_workflow_progress(app, initial_state):
                if event_type == "progress":
                    yield event_data
                elif event_type == "state":
                    result_state = event_data

            # Validate workflow completed successfully
            result_state = validate_workflow_state(result_state)

            # Generate and yield metadata events
            for event in generate_metadata_events(result_state, area_id, company_id):
                yield event

            # Stream LLM response and save message
            if tts:
                # TTS enabled: use the TTS streaming workflow
                llm_stream = stream_llm_response(
                    state=result_state,
                    llm_provider=self.llm_provider,
                    message_service=self.message_service,
                    db=db
                )

                # Use injected TTS provider if available (connection reuse)
                async for event in stream_with_tts(
                    llm_stream,
                    tts_provider=self.tts_provider,
                    debug=True
                ):
                    yield event
            else:
                # TTS disabled: stream text only
                async for chunk_event in stream_llm_response(
                    state=result_state,
                    llm_provider=self.llm_provider,
                    message_service=self.message_service,
                    db=db
                ):
                    yield chunk_event

            # Send completion signal
            yield {
                "type": "complete",
                "status": "success"
            }

        except ValueError as e:
            # Workflow validation errors
            logger.error(f"Workflow error: {e}")
            yield {
                "type": "error",
                "message": str(e),
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": str(e)
                }
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error in streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": "Unexpected error",
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error inesperado: {str(e)}"
                }
            }

    async def process_rag_query_n8n(self, user_id: int, message: str, company_id: int, area_id: int, db: Session, created_at: str, chat_id: int, request_timezone: str = None) -> Dict[str, Any]:
        """
        Proceso RAG completo sin streaming (para n8n): embeddings → search → LLM → response completa
        Uses pre-compiled LangGraph workflow for modular processing with complete response.
        """
        try:
            # Get pre-compiled workflow
            app = get_compiled_rag_workflow()

            # Build initial state
            initial_state = build_initial_state(
                user_id=user_id,
                message=message,
                company_id=company_id,
                area_id=area_id,
                created_at=created_at,
                chat_id=str(chat_id),
                request_timezone=request_timezone
            )

            # Execute workflow without streaming
            result_state = await execute_workflow_n8n(app, initial_state)

            # Validate workflow completed successfully
            result_state = validate_workflow_state(result_state)

            # Generate complete LLM response and save
            assistant_response = await generate_complete_llm_response(
                state=result_state,
                llm_nonstreaming_provider=self.llm_nonstreaming_provider,
                message_service=self.message_service,
                db=db
            )

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except ValueError as e:
            # Workflow validation errors
            logger.error(f"Workflow error: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": str(e)
                }
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Error in process_rag_query_n8n: {e}", exc_info=True)
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta: {str(e)}"
                }
            }

    async def process_rag_query_n8n_anonymous(self, user_anonymous_id: int, message: str, company_id: int, area_id: int, db: Session, created_at: str, chat_anonymous_id: int, rag_query: str, request_timezone: str = None) -> Dict[str, Any]:
        """
        Proceso RAG completo sin streaming para chats anónimos (para n8n): embeddings → search → LLM → response completa
        Uses pre-compiled LangGraph anonymous workflow for modular processing with complete response.
        """
        try:
            # Get pre-compiled anonymous workflow
            app = get_compiled_rag_anonymous_workflow()

            # Build initial state for anonymous chat
            initial_state = build_initial_state_anonymous(
                user_anonymous_id=user_anonymous_id,
                message=message,
                company_id=company_id,
                area_id=area_id,
                created_at=created_at,
                rag_query=rag_query,
                chat_anonymous_id=str(chat_anonymous_id),
                request_timezone=request_timezone
            )

            # Execute workflow without streaming
            result_state = await execute_workflow_n8n(app, initial_state)

            # Validate workflow completed successfully
            result_state = validate_workflow_state_anonymous(result_state)

            # Generate complete LLM response and save for anonymous chat
            assistant_response = await generate_complete_llm_response_anonymous(
                state=result_state,
                llm_nonstreaming_provider=self.llm_nonstreaming_provider,
                message_service=self.message_service,
                db=db
            )

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except ValueError as e:
            # Workflow validation errors
            logger.error(f"Anonymous workflow error: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": str(e)
                }
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Error in process_rag_query_n8n_anonymous: {e}", exc_info=True)
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta anónima: {str(e)}"
                }
            }

    async def process_llm_only_n8n(self, user_id: int, message: str, company_id: int, db: Session, created_at: str, chat_id: int, system_behavior: str = None, custom_llm: str = None, request_timezone: str = None, use_guidelines: bool = True, store_messages: bool = True) -> Dict[str, Any]:
        """
        Proceso LLM-only sin streaming (para n8n): LLM → response completa (sin embeddings, sin vector stores, sin state builder, sin query rewriter)
        Uses pre-compiled LangGraph LLM-only workflow for modular processing.
        """
        try:
            # Get pre-compiled LLM-only workflow
            app = get_compiled_llm_only_workflow()

            # Build initial state
            initial_state = build_llm_only_initial_state(
                user_id=user_id,
                message=message,
                company_id=company_id,
                created_at=created_at,
                chat_id=str(chat_id),
                system_behavior=system_behavior,
                request_timezone=request_timezone,
                use_guidelines=use_guidelines,
                store_messages=store_messages,
                custom_llm=custom_llm
            )

            # Execute workflow without streaming
            result_state = await execute_workflow_n8n(app, initial_state)

            # Validate workflow completed successfully
            result_state = validate_llm_only_workflow_state(result_state)

            # Generate complete LLM response and save
            assistant_response = await generate_complete_llm_only_response(
                state=result_state,
                llm_only_provider=self.llm_only_provider,
                message_service=self.message_service,
                db=db
            )

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except ValueError as e:
            # Workflow validation errors
            logger.error(f"Workflow error: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": str(e)
                }
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Error in process_llm_only_n8n: {e}", exc_info=True)
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta: {str(e)}"
                }
            }

    async def process_llm_only_n8n_anonymous(self, user_anonymous_id: int, message: str, company_id: int, db: Session, created_at: str, chat_anonymous_id: int, system_behavior: str = None, custom_llm: str = None, request_timezone: str = None, use_guidelines: bool = True, store_messages: bool = True) -> Dict[str, Any]:
        """
        Proceso LLM-only sin streaming para chats anónimos (para n8n): LLM → response completa (sin embeddings, sin vector stores, sin state builder, sin query rewriter)
        Uses pre-compiled LangGraph LLM-only anonymous workflow for modular processing.
        """
        try:
            # Get pre-compiled LLM-only anonymous workflow
            app = get_compiled_llm_only_anonymous_workflow()

            # Build initial state for anonymous chat
            initial_state = build_llm_only_initial_state_anonymous(
                user_anonymous_id=user_anonymous_id,
                message=message,
                company_id=company_id,
                created_at=created_at,
                chat_anonymous_id=str(chat_anonymous_id),
                system_behavior=system_behavior,
                request_timezone=request_timezone,
                use_guidelines=use_guidelines,
                store_messages=store_messages,
                custom_llm=custom_llm
            )

            # Execute workflow without streaming
            result_state = await execute_workflow_n8n(app, initial_state)

            # Validate workflow completed successfully
            result_state = validate_llm_only_workflow_state_anonymous(result_state)

            # Generate complete LLM response and save for anonymous chat
            assistant_response = await generate_complete_llm_only_response_anonymous(
                state=result_state,
                llm_only_provider=self.llm_only_provider,
                message_service=self.message_service,
                db=db
            )

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except ValueError as e:
            # Workflow validation errors
            logger.error(f"Anonymous LLM-only workflow error: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": str(e)
                }
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Error in process_llm_only_n8n_anonymous: {e}", exc_info=True)
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta anónima: {str(e)}"
                }
            }

    async def process_vlm_query_stream(
        self,
        user_id: int,
        message: str,
        company_id: int,
        area_id: int,
        db,
        created_at: str,
        filenames: list,
        chat_id: str = None,
        request_timezone: str = None,
        vlm_mode: str = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        VLM streaming: build attachment keys → save message → call VLM → stream response
        Uses pre-compiled VLM LangGraph workflow.
        """
        try:
            app = get_compiled_vlm_workflow()

            initial_state = build_vlm_initial_state(
                user_id=user_id,
                message=message,
                company_id=company_id,
                area_id=area_id,
                created_at=created_at,
                filenames=filenames,
                chat_id=chat_id,
                request_timezone=request_timezone,
                vlm_mode=vlm_mode,
            )

            yield {"type": "progress", "message": "Analizando imágenes..."}

            result_state = None
            async for state in app.astream(initial_state, stream_mode="values"):
                result_state = state

            result_state = validate_vlm_workflow_state(result_state)

            for event in generate_metadata_events(result_state, area_id, company_id):
                yield event

            async for chunk_event in stream_vlm_response(
                state=result_state,
                vlm_provider=self.vlm_provider,
                message_service=self.message_service,
                db=db,
            ):
                yield chunk_event

            yield {"type": "complete", "status": "success"}

        except ValueError as e:
            logger.error(f"VLM workflow error: {e}")
            yield {
                "type": "error",
                "message": str(e),
                "result": {"idTipoMensaje": 1, "mensaje": str(e)}
            }
        except Exception as e:
            logger.error(f"Unexpected error in VLM streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": "Unexpected error",
                "result": {"idTipoMensaje": 1, "mensaje": f"Error inesperado: {str(e)}"}
            }
