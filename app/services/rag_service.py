from typing import Tuple, List, Dict, Any, AsyncGenerator, Optional
import logging
import json
import re
import time
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.llm_nonstreaming_port import LLMNonStreamingPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.domain.ports.state_builder import StateBuilderPort
from app.domain.ports.query_rewriter import QueryRewriterPort
from app.core.config import settings
from app.infrastructure.task_decomposition.task_generator import TaskGenerator
from app.infrastructure.api_clients.api_client import httpx_get, httpx_post
from app.services.message_service import MessageService
from app.services.ia_config_service import IaConfigService
from app.infrastructure.repositories.chat_repository import ChatRepository
from app.infrastructure.llm.model_factory import ModelConfigFactory
from app.utils.query_utils import clean_user_query
from app.utils.search_utils import build_context_from_search_results
from app.utils.llm_utils import generate_text_stream_with_validation, generate_text_with_validation, generate_text_llm_only_with_validation
from app.utils.time_utils import format_timestamp_with_timezone
from app.workflows.rag_workflow import get_compiled_rag_workflow, RAGState

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
        state_builder: StateBuilderPort,
        query_rewriter: QueryRewriterPort,
        orchestrator: Optional[QueryAnalysisPort] = None,
        llm_nonstreaming_provider: LLMNonStreamingPort = None,
        llm_only_provider: LLMNonStreamingPort = None
    ):
        """
        Initialize RagService with all dependencies injected.

        Args:
            embeddings_provider: Port for generating embeddings
            vectorstore: Port for vector database operations
            llm_provider: Port for LLM operations (streaming)
            message_service: Service for managing chat messages in DynamoDB
            ia_config_service: Service for loading IA area configuration
            state_builder: Service for building query state from conversation history
            query_rewriter: Service for rewriting queries based on state
            orchestrator: Optional port for query analysis and task decomposition
            llm_nonstreaming_provider: Optional port for non-streaming LLM operations (for n8n RAG mode)
            llm_only_provider: Optional port for non-streaming LLM operations in LLM-only mode (no RAG)
        """
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
        self.llm_provider = llm_provider
        self.message_service = message_service
        self.ia_config_service = ia_config_service
        self.state_builder = state_builder
        self.query_rewriter = query_rewriter
        self.orchestrator = orchestrator
        self.llm_nonstreaming_provider = llm_nonstreaming_provider
        self.llm_only_provider = llm_only_provider

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


    async def process_rag_query_stream(self, user_id: int, message: str, company_id: int, area_id: int, db: Session, created_at: str, chat_id: str = None, request_timezone: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Proceso RAG completo con streaming: embeddings → search → LLM streaming → response
        Uses pre-compiled LangGraph workflow for modular processing.
        """
        # Get pre-compiled workflow (compiled once at startup)
        app = get_compiled_rag_workflow()

        # Initialize state with only request data (no dependencies)
        initial_state: RAGState = {
            # Input parameters
            "user_id": user_id,
            "message": message,
            "company_id": company_id,
            "area_id": area_id,
            "created_at": created_at,
            "chat_id": chat_id,
            "request_timezone": request_timezone,
            # Processing state (will be populated by workflow)
            "cleaned_message": None,
            "conversation_history": [],
            "state_builder_result": None,
            "query_rewriter_result": None,
            "rag_config": None,
            "new_chat_created": False,
            "new_chat_titulo": None,
            "new_chat_timestamp": None,
            "query_for_search": None,
            "query_embedding": None,
            "search_results": None,
            "context_text": None,
            "conversation_history_for_prompt": [],
            "rag_prompt": None,
            "assistant_timestamp": None,
            "assistant_timestamp_ms": None,
            "utc_formatted": None,
            "local_formatted": None,
            # Error handling
            "error": None,
            "should_stop": False
        }

        # Execute workflow
        try:
            result_state = await app.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            yield {
                "type": "error",
                "message": "Workflow execution failed",
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error en el procesamiento: {str(e)}"
                }
            }
            return

        # Check if workflow encountered an error
        if result_state.get("should_stop", False):
            error_msg = result_state.get("error", "Unknown error")
            logger.error(f"Workflow stopped with error: {error_msg}")
            yield {
                "type": "error",
                "message": error_msg,
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": error_msg
                }
            }
            return

        # Extract processed state
        chat_id = result_state["chat_id"]
        rag_config = result_state["rag_config"]
        cleaned_message = result_state["cleaned_message"]
        rag_prompt = result_state["rag_prompt"]
        conversation_history_for_prompt = result_state["conversation_history_for_prompt"]
        assistant_timestamp = result_state["assistant_timestamp"]
        assistant_timestamp_ms = result_state["assistant_timestamp_ms"]
        utc_formatted = result_state["utc_formatted"]
        local_formatted = result_state["local_formatted"]

        # Yield metadata early (before processing) so frontend can show "Pensando..." placeholder
        yield {
            "type": "metadata",
            "chat_id": chat_id
        }

        # If a new chat was created, send the chat object to the frontend
        if result_state.get("new_chat_created", False):
            yield {
                "type": "chat_created",
                "chat": {
                    "ID_CHAT": chat_id,
                    "ID_AREA": area_id,
                    "ID_EMPRESA": company_id,
                    "TITULO": result_state["new_chat_titulo"],
                    "ULTIMO_MENSAJE_FECHA": result_state["new_chat_timestamp"],
                    "ID_ESTADO_REGISTRO": 1
                }
            }

        # Send assistant_metadata BEFORE starting LLM streaming
        yield {
            "type": "assistant_metadata",
            "sender": 1,
            "created_at": assistant_timestamp
        }

        # Accumulate assistant response chunks
        assistant_response = ""
        first_chunk_sent = False

        # Stream the LLM response with role behavior and conversation history
        async for chunk in generate_text_stream_with_validation(
            llm_provider=self.llm_provider,
            model_id=rag_config['config']['LLM_MODEL'],
            prompt=rag_prompt,
            max_tokens=rag_config['config']['LLM_MAX_TOKENS'],
            temperature=rag_config['config']['LLM_TEMPERATURE'],
            top_p=rag_config['config']['LLM_TOP_P'],
            role_behavior=rag_config['config']['ROLE_BEHAVIOR'],
            messages=conversation_history_for_prompt if conversation_history_for_prompt else None,
            request_timezone=request_timezone,
            utc_formatted=utc_formatted,
            local_formatted=local_formatted
        ):
            assistant_response += chunk

            # Update chat last message date when first chunk with content is sent
            if not first_chunk_sent and chunk.strip():
                chat_repo = ChatRepository(db)
                await asyncio.to_thread(
                    chat_repo.update_ultimo_mensaje_fecha,
                    chat_id
                )
                first_chunk_sent = True

            yield {
                "type": "chunk",
                "content": chunk
            }

        # Save assistant message to DynamoDB before completion signal
        if chat_id and assistant_response and assistant_timestamp:
            try:
                await self.message_service.create_message(
                    chat_id=chat_id,
                    created_at=assistant_timestamp,
                    sender=1,  # 1 = assistant
                    message=assistant_response
                )
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")
                yield {
                    "type": "error",
                    "message": "Failed to save assistant message",
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El mensaje de la IA no pudo guardarse correctamente"
                    }
                }
                return

        # Final completion signal
        yield {
            "type": "complete",
            "status": "success"
        }

    async def process_rag_query_n8n(self, user_id: int, message: str, company_id: int, area_id: int, db: Session, created_at: str, chat_id: int, request_timezone: str = None) -> Dict[str, Any]:
        """
        Proceso RAG completo sin streaming (para n8n): embeddings → search → LLM → response completa
        Retorna directamente la respuesta completa del LLM con resultado estructurado.
        """
        try:
            # Validate non-streaming provider is available
            if not self.llm_nonstreaming_provider:
                logger.error("Non-streaming LLM provider not configured")
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "Servicio de generación no configurado correctamente"
                    }
                }

            # Validate inputs
            if not message or not message.strip():
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El mensaje no puede estar vacío"
                    }
                }
            if not user_id:
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "ID de usuario requerido"
                    }
                }
            if not company_id:
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "ID de empresa requerido"
                    }
                }
            if not area_id:
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "ID de área requerido"
                    }
                }

            # Get last messages from chat history
            conversation_history = []
            try:
                messages = await self.message_service.get_last_n_messages(
                    chat_id=f"chat-{chat_id}",
                    n=20
                )

                if messages:
                    # Build and collapse in one pass
                    collapsed = []
                    for msg in messages:
                        role = "user" if msg.sender == 0 else "assistant"
                        if not collapsed or collapsed[-1]["role"] != role:
                            collapsed.append({"role": role, "content": msg.message})
                        else:
                            collapsed[-1]["content"] = msg.message

                    # Find boundaries
                    start = next((i for i, m in enumerate(collapsed) if m["role"] == "user"), -1)
                    end = next((i for i in range(len(collapsed)-1, -1, -1)
                               if collapsed[i]["role"] == "assistant"), -1)

                    # Validate and slice
                    if start >= 0 and end > start:
                        segment = collapsed[start:end+1]

                        # Quick alternating check (should be true due to collapse)
                        if all(segment[i]["role"] != segment[i+1]["role"]
                               for i in range(len(segment)-1)):

                            # Trim to context window (keep newest)
                            MAX_CTX = 16
                            if len(segment) > MAX_CTX:
                                trim = len(segment) - MAX_CTX
                                if segment[trim]["role"] == "assistant":
                                    trim -= 1
                                segment = segment[max(0, trim):]

                            # Final check
                            if segment and segment[0]["role"] == "user":
                                conversation_history = segment

            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {e}, continuing without history")
                conversation_history = []

            cleaned_message = clean_user_query(message)

            # Build query state and rewrite query if conversation history exists
            state_builder_result = None
            query_rewriter_result = None
            if conversation_history:
                # Build query state for RAG (using last 6 messages, replace assistant content)
                try:
                    conversation_for_state_building = conversation_history[-6:] if len(conversation_history) >= 6 else conversation_history
                    # Replace assistant messages content with placeholder (Bedrock doesn't allow empty content)
                    conversation_for_state_building = [
                        {**msg, "content": "assistant message"} if msg["role"] == "assistant" else msg
                        for msg in conversation_for_state_building
                    ]

                    state_builder_result = await self.state_builder.build_query_state(
                        user_query=cleaned_message,
                        conversation_history=conversation_for_state_building
                    )
                    logger.info(f"State builder result: {state_builder_result}")
                except Exception as e:
                    logger.warning(f"Failed to build query state: {e}, continuing without state building")
                    state_builder_result = None

                # Rewrite query based on state builder result
                if state_builder_result:
                    try:
                        query_rewriter_result = await self.query_rewriter.rewrite_query(
                            user_query=cleaned_message,
                            state=state_builder_result
                        )
                        logger.info(f"Query rewriter result: {query_rewriter_result}")
                    except Exception as e:
                        logger.warning(f"Failed to rewrite query: {e}, continuing without query rewriting")
                        query_rewriter_result = None

            # Load IA area RAG configuration
            rag_config = await self.ia_config_service.get_ia_area_config_rag(db, company_id, area_id)
            logger.info(f"Retrieved RAG config: {rag_config}")

            # Save user message to DynamoDB
            try:
                await self.message_service.create_message(
                    chat_id=chat_id,
                    created_at=created_at,
                    sender=0,
                    message=cleaned_message
                )
            except Exception as e:
                logger.error(f"Failed to save user message: {e}")
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El mensaje del usuario no pudo guardarse correctamente"
                    }
                }

            # Determine query to use for embedding and search
            query_for_search = cleaned_message
            if query_rewriter_result and query_rewriter_result.get("needs_rewrite", False):
                rewritten_query = query_rewriter_result.get("rewritten_query", "").strip()
                if rewritten_query:
                    query_for_search = rewritten_query
                    logger.info(f"Using rewritten query for search: {query_for_search}")

            # Generate embedding for the query
            query_embedding = await self.embeddings_provider.embed(query_for_search)

            # Search vector database
            search_results = await self.vectorstore.search_in_collection_hybrid(
                company_id=company_id,
                area_id=area_id,
                query_text=query_for_search,
                query_vector=query_embedding,
                top_k=rag_config['config']['RAG_TOP_K_RESULTS'],
                similarity_threshold=rag_config['config']['RAG_SIMILARITY_THRESHOLD'],
                alpha=rag_config['config']['RAG_ALPHA'],
                general_area=rag_config.get('general_area')
            )

            # Prepare context text for LLM
            context_text = build_context_from_search_results(search_results)

            # Select conversation history for LLM prompt
            conversation_history_for_prompt = []
            if conversation_history and query_rewriter_result:
                needs_rewrite = query_rewriter_result.get("needs_rewrite", False)
                summary_intent = query_rewriter_result.get("is_summary_request", False)
                messages_to_use = 0
                if needs_rewrite and summary_intent:
                    messages_to_use = 16
                elif needs_rewrite:
                    messages_to_use = 8
                elif summary_intent:
                    messages_to_use = 16
                if messages_to_use > 0:
                    conversation_history_for_prompt = conversation_history[-messages_to_use:] if len(conversation_history) >= messages_to_use else conversation_history

            # Build RAG prompt
            model_config = ModelConfigFactory.get_model_config(rag_config['config']['LLM_MODEL'])
            rag_prompt = model_config.build_rag_prompt(cleaned_message, context_text)

            # Generate timestamp for assistant message and format timestamps
            assistant_timestamp_ms = int(time.time() * 1000)
            user_timestamp_ms = int(created_at)
            if assistant_timestamp_ms < user_timestamp_ms:
                assistant_timestamp_ms = user_timestamp_ms + 1000

            # Format timestamp with timezone
            utc_formatted, local_formatted = format_timestamp_with_timezone(
                assistant_timestamp_ms,
                request_timezone or "America/Lima"
            )

            # Generate complete response using non-streaming provider with validation
            assistant_response = await generate_text_with_validation(
                llm_provider=self.llm_nonstreaming_provider,
                model_id=rag_config['config']['LLM_MODEL'],
                prompt=rag_prompt,
                max_tokens=rag_config['config']['LLM_MAX_TOKENS'],
                temperature=rag_config['config']['LLM_TEMPERATURE'],
                top_p=rag_config['config']['LLM_TOP_P'],
                role_behavior=rag_config['config']['ROLE_BEHAVIOR'],
                messages=conversation_history_for_prompt if conversation_history_for_prompt else None,
                request_timezone=request_timezone,
                utc_formatted=utc_formatted,
                local_formatted=local_formatted
            )

            # Update chat last message date
            chat_repo = ChatRepository(db)
            await asyncio.to_thread(
                chat_repo.update_ultimo_mensaje_fecha,
                chat_id
            )

            # Use the timestamp generated earlier
            assistant_timestamp = str(assistant_timestamp_ms)

            # Save assistant message to DynamoDB
            if assistant_response and assistant_timestamp:
                try:
                    await self.message_service.create_message(
                        chat_id=chat_id,
                        created_at=assistant_timestamp,
                        sender=1,  # 1 = assistant
                        message=assistant_response
                    )
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")
                    return {
                        "result": {
                            "idTipoMensaje": 1,
                            "mensaje": "El mensaje de la IA no pudo guardarse correctamente"
                        }
                    }

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except Exception as e:
            logger.error(f"Error in process_rag_query_n8n: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta: {str(e)}"
                }
            }

    async def process_llm_only_n8n(self, user_id: int, message: str, company_id: int, db: Session, created_at: str, chat_id: int, system_behavior: str, request_timezone: str = None, use_guidelines: bool = True, store_messages: bool = True) -> Dict[str, Any]:
        """
        Proceso LLM-only sin streaming (para n8n): LLM → response completa (sin embeddings, sin vector stores, sin state builder, sin query rewriter)
        Retorna directamente la respuesta completa del LLM con resultado estructurado.
        """
        try:
            # Validate LLM-only provider is available
            if not self.llm_only_provider:
                logger.error("LLM-only provider not configured")
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "Servicio de generación LLM-only no configurado correctamente"
                    }
                }

            # Validate inputs
            if not message or not message.strip():
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El mensaje no puede estar vacío"
                    }
                }
            if not user_id:
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "ID de usuario requerido"
                    }
                }
            if not company_id:
                return {
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "ID de empresa requerido"
                    }
                }

            # Get last messages from chat history
            conversation_history = []
            try:
                messages = await self.message_service.get_last_n_messages(
                    chat_id=f"chat-{chat_id}",
                    n=20
                )

                if messages:
                    # Build and collapse in one pass
                    collapsed = []
                    for msg in messages:
                        role = "user" if msg.sender == 0 else "assistant"
                        if not collapsed or collapsed[-1]["role"] != role:
                            collapsed.append({"role": role, "content": msg.message})
                        else:
                            collapsed[-1]["content"] = msg.message

                    # Find boundaries
                    start = next((i for i, m in enumerate(collapsed) if m["role"] == "user"), -1)
                    end = next((i for i in range(len(collapsed)-1, -1, -1)
                               if collapsed[i]["role"] == "assistant"), -1)

                    # Validate and slice
                    if start >= 0 and end > start:
                        segment = collapsed[start:end+1]

                        # Quick alternating check (should be true due to collapse)
                        if all(segment[i]["role"] != segment[i+1]["role"]
                               for i in range(len(segment)-1)):

                            # Trim to context window (keep newest)
                            MAX_CTX = 8
                            if len(segment) > MAX_CTX:
                                trim = len(segment) - MAX_CTX
                                if segment[trim]["role"] == "assistant":
                                    trim -= 1
                                segment = segment[max(0, trim):]

                            # Final check
                            if segment and segment[0]["role"] == "user":
                                conversation_history = segment

            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {e}, continuing without history")
                conversation_history = []

            cleaned_message = clean_user_query(message)

            # Load IA area RAG configuration (company-level only, no area required)
            rag_config = await self.ia_config_service.get_ia_area_config_rag_no_area(db, company_id)
            logger.info(f"Retrieved RAG config (no area): {rag_config}")

            # Save user message to DynamoDB (if store_messages is enabled)
            if store_messages:
                try:
                    await self.message_service.create_message(
                        chat_id=chat_id,
                        created_at=created_at,
                        sender=0,
                        message=cleaned_message
                    )
                except Exception as e:
                    logger.error(f"Failed to save user message: {e}")
                    return {
                        "result": {
                            "idTipoMensaje": 1,
                            "mensaje": "El mensaje del usuario no pudo guardarse correctamente"
                        }
                    }
            else:
                logger.info("Skipping user message storage (store_messages=False)")

            # Use all available conversation history for LLM prompt (up to 8 messages)
            conversation_history_for_prompt = conversation_history if conversation_history else []

            # Generate timestamp for assistant message and format timestamps
            assistant_timestamp_ms = int(time.time() * 1000)
            user_timestamp_ms = int(created_at)
            if assistant_timestamp_ms < user_timestamp_ms:
                assistant_timestamp_ms = user_timestamp_ms + 1000

            # Format timestamp with timezone
            utc_formatted, local_formatted = format_timestamp_with_timezone(
                assistant_timestamp_ms,
                request_timezone or "America/Lima"
            )

            # Generate complete response using LLM-only provider with validation
            # This provider uses a system prompt optimized for conversational AI without RAG
            assistant_response = await generate_text_llm_only_with_validation(
                llm_provider=self.llm_only_provider,
                model_id=rag_config['config']['LLM_MODEL'],
                prompt=cleaned_message,
                max_tokens=rag_config['config']['LLM_MAX_TOKENS'],
                temperature=rag_config['config']['LLM_TEMPERATURE'],
                top_p=rag_config['config']['LLM_TOP_P'],
                role_behavior=system_behavior,
                messages=conversation_history_for_prompt if conversation_history_for_prompt else None,
                request_timezone=request_timezone,
                utc_formatted=utc_formatted,
                local_formatted=local_formatted,
                use_guidelines=use_guidelines
            )

            # Update chat last message date
            chat_repo = ChatRepository(db)
            await asyncio.to_thread(
                chat_repo.update_ultimo_mensaje_fecha,
                chat_id
            )

            # Use the timestamp generated earlier
            assistant_timestamp = str(assistant_timestamp_ms)

            # Save assistant message to DynamoDB (if store_messages is enabled)
            if store_messages:
                if assistant_response and assistant_timestamp:
                    try:
                        await self.message_service.create_message(
                            chat_id=chat_id,
                            created_at=assistant_timestamp,
                            sender=1,  # 1 = assistant
                            message=assistant_response
                        )
                    except Exception as e:
                        logger.error(f"Failed to save assistant message: {e}")
                        return {
                            "result": {
                                "idTipoMensaje": 1,
                                "mensaje": "El mensaje de la IA no pudo guardarse correctamente"
                            }
                        }
            else:
                logger.info("Skipping assistant message storage (store_messages=False)")

            # Return successful response
            return {
                "response": assistant_response,
                "result": {
                    "idTipoMensaje": 2,
                    "mensaje": "Respuesta generada correctamente"
                }
            }

        except Exception as e:
            logger.error(f"Error in process_rag_query_n8n: {e}")
            return {
                "result": {
                    "idTipoMensaje": 1,
                    "mensaje": f"Error al procesar la consulta: {str(e)}"
                }
            }
