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
from app.domain.ports.comparator_port import ComparatorPort

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
        comparator: ComparatorPort,
        orchestrator: Optional[QueryAnalysisPort] = None,
        llm_nonstreaming_provider: LLMNonStreamingPort = None,
        llm_only_provider: LLMNonStreamingPort = None,
        tts_provider = None  # Optional: inject for connection reuse
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
        self.comparator = comparator
        self.orchestrator = orchestrator
        self.llm_nonstreaming_provider = llm_nonstreaming_provider
        self.llm_only_provider = llm_only_provider
        self.tts_provider = tts_provider

    # async def agent_orchestrator_stream(self, user_id: int, user: str, message: str, company_id: int, company: str, area_id: int, area: str, id_ia_area: int, db: Session, top_k: int = None, similarity_threshold: float = None, alpha: float = None, temperature: float = None, max_tokens: int = None, external_token: str = None):
    #     """
    #     Analyze user query using the agent orchestrator model to determine workflow requirements.
    #     Enhanced version that accepts all process_rag_query_stream parameters for complete context.
    #     """
    #     try:
    #         # Validate inputs
    #         if not message or not message.strip():
    #             raise ValueError("Message cannot be empty")
    #         if not user_id:
    #             raise ValueError("User ID is required")
    #         if not company or not company.strip():
    #             raise ValueError("Company ID is required and cannot be empty")
    #         if not area or not area.strip():
    #             raise ValueError("Area is required and cannot be empty")
    #         if id_ia_area is None:
    #             raise ValueError("ID IA Area is required and cannot be empty")
    #         if not external_token or not external_token.strip():
    #             raise ValueError("External token is required and cannot be empty")
    #
    #         # Use injected orchestrator
    #         if not self.orchestrator:
    #             raise ValueError("Orchestrator not configured for this service instance")
    #
    #         orchestrator = self.orchestrator
    #
    #         # Load IA area role behavior configuration
    #         role_behavior = await self.ia_config_service.get_ia_area_config(db, id_ia_area)
    #
    #         # Clean user query
    #         user_query = clean_user_query(message)
    #
    #         # Define available APIs (this could be loaded from config)
    #         available_apis = [
    #             {
    #                 "method": "GET",
    #                 "endpoint": "/bdt/talent/list",
    #                 "description": "Table: talents, Columns: idTalento, nombres, apellidoPaterno, apellidoMaterno, imagen, puesto, pais, ciudad, idModalidadFacturacion, montoInicialPlanilla, montoFinalPlanilla, montoInicialRxH, montoFinalRxH, moneda, estrellas, esFavorito, idMonedaPlan, idMonedaRxh",
    #                 "params": {
    #                     "nPag": { "type": "integer", "required": False },
    #                     "search": { "type": "string", "required": False },
    #                     "techAbilities": { "type": "string", "required": False },
    #                     "idEnglishLevel": { "type": "integer", "required": False },
    #                     "idTalentCollection": { "type": "integer", "required": False }
    #                 }
    #             }
    #         ]
    #
    #         # Call orchestrator to analyze the query
    #         analysis = await orchestrator.analyze_query(user_query, available_apis)
    #
    #         # Generate tasks from analysis
    #         tasks = TaskGenerator.generate_tasks_from_analysis(analysis, available_apis, user_query)
    #
    #         # Execute tasks sequentially
    #         query_embedding = None
    #         context_text = ""
    #         execution_failed = False
    #
    #         for task in tasks:
    #             if execution_failed:
    #                 break
    #             if task.get("action") == "embedding":
    #                 try:
    #                     query_text = task.get("input", user_query)
    #                     query_embedding = await self.embeddings_provider.embed(query_text)
    #                 except Exception as e:
    #                     execution_failed = True
    #                     break  # Stop execution if embedding fails
    #
    #             elif task.get("action") == "retrieval":
    #                 try:
    #                     if query_embedding is None:
    #                         # Generate embedding if not already done
    #                         query_embedding = await self.embeddings_provider.embed(user_query)
    #
    #                     # Use semantic_query from analysis if available, otherwise use user_query
    #                     semantic_query = analysis.get("semantic_query", "").strip() if analysis.get("semantic_query") else user_query
    #
    #                     # Perform hybrid search using vectorstore directly
    #                     search_results = await self.vectorstore.search_in_collection_hybrid(
    #                         company_id=company_id,
    #                         area_id=area_id,
    #                         query_text=semantic_query,
    #                         query_vector=query_embedding,
    #                         top_k=top_k,
    #                         similarity_threshold=similarity_threshold,
    #                         alpha=alpha
    #                     )
    #
    #                     # Build context from retrieved documents using utility function
    #                     context_text = build_context_from_search_results(search_results)
    #                 except Exception as e:
    #                     execution_failed = True
    #                     break  # Stop execution if retrieval fails
    #
    #             elif task.get("action") == "api_call":
    #                 try:
    #                     method = task.get("method", "GET").upper()
    #                     if method == "GET":
    #                         api_result = await httpx_get(task.get("endpoint", ""), external_token, task.get("params", {}))
    #                     else:
    #                         api_result = await httpx_post(task.get("endpoint", ""), external_token, task.get("params", {}))
    #
    #                     # Add API result to context only if there's actual data
    #                     if api_result.get("success") and api_result.get("data"):
    #                         api_data = json.dumps(api_result['data'])
    #                         if api_data and api_data.strip() not in ["{}", "[]", "null"]:
    #                             # Format API call result with metadata
    #                             api_context_parts = []
    #                             api_context_parts.append(f"API Call:")
    #                             api_context_parts.append(f"Endpoint: {task.get('endpoint', 'N/A')}")
    #                             api_context_parts.append(f"Params: {json.dumps(task.get('params', {}))}")
    #                             api_context_parts.append(f"Response:\n{api_data}")
    #
    #                             formatted_api_context = "\n".join(api_context_parts)
    #
    #                             # Add separator if context already has content
    #                             if context_text:
    #                                 context_text += "\n\n"
    #                             context_text += formatted_api_context
    #
    #                 except Exception as e:
    #                     execution_failed = True
    #                     break  # Stop execution if API call fails
    #
    #             elif task.get("action") == "llm_response":
    #                 try:
    #                     # Check if we have any context at all
    #                     if not context_text or context_text.strip() == "":
    #                         # No context available - return predefined message
    #                         yield {
    #                             "type": "chunk",
    #                             "content": NO_CONTEXT_MESSAGE
    #                         }
    #                         break
    #
    #                     # Prepare the query, checking for format requirements
    #                     query_to_use = user_query
    #                     task_format = task.get("format")
    #                     if task_format:
    #                         if task_format.lower() == "list":
    #                             query_to_use = f"{user_query}. IMPORTANT: Format the list items as a compact markdown list in a single line per item, including only the key and necessary information for clear understanding. If there is additional text content after the list, continue with it as plain text below the list."
    #                         elif task_format.lower() == "table":
    #                             query_to_use = f"{user_query}. IMPORTANT: Provide your response in the same language as the question. If explanation or summary is needed, include it briefly before the table. Then present the tabular data as a well-structured markdown table, showing ALL rows and ALL columns without adding empty or duplicate rows. Convert headers to natural language in the same language as the question, and ensure the table is properly aligned and easy to read. If there is additional text content after the table, continue with it as plain text below the table."
    #
    #                     # Build prompt with context (we know context_text exists here)
    #                     model_config = self.llm_provider.get_model_config()
    #                     prompt = model_config.build_rag_prompt(query_to_use, context_text)
    #
    #                     # Use provided parameters or fall back to environment defaults
    #                     llm_temperature = temperature if temperature is not None else settings.llm_temperature
    #                     llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    #
    #                     # Track if assistant metadata has been sent
    #                     agent_timestamp_sent = False
    #
    #                     # Stream response using utility function for consistent stop reason handling
    #                     async for chunk in generate_text_stream_with_validation(
    #                         llm_provider=self.llm_provider,
    #                         prompt=prompt,
    #                         max_tokens=llm_max_tokens,
    #                         temperature=llm_temperature,
    #                         role_behavior=role_behavior
    #                     ):
    #                         # Send agent metadata on first chunk
    #                         if not agent_timestamp_sent:
    #                             agent_timestamp = str(int(time.time() * 1000))
    #                             yield {
    #                                 "type": "assistant_metadata",
    #                                 "sender": 2,  # 2 = agent
    #                                 "created_at": agent_timestamp
    #                             }
    #                             agent_timestamp_sent = True
    #
    #                         # Yield each chunk for streaming (same format as process_rag_query_stream)
    #                         yield {
    #                             "type": "chunk",
    #                             "content": chunk
    #                         }
    #
    #                 except Exception as e:
    #                     execution_failed = True
    #                     break  # Stop execution if LLM response fails
    #
    #         # Send completion signal
    #         yield {
    #             "type": "complete",
    #             "status": "success"
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Error in agent_orchestrator_stream: {e}")
    #         yield {
    #             "type": "error",
    #             "content": f"An error occurred: {str(e)}"
    #         }


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
