from typing import Tuple, List, Dict, Any, AsyncGenerator, Optional
import logging
import json
import re
import time
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.core.config import settings
from app.infrastructure.task_decomposition.task_generator import TaskGenerator
from app.infrastructure.api_clients.api_client import httpx_get, httpx_post
from app.services.message_service import MessageService
from app.infrastructure.recontextualizer.aws_bedrock_provider import QueryRecontextualizer
from app.infrastructure.repositories.chat_repository import ChatRepository

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

    def __init__(self, embeddings_provider: EmbeddingsPort, vectorstore: VectorStorePort, llm_provider: LLMPort, orchestrator: QueryAnalysisPort = None):
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
        self.llm_provider = llm_provider
        self.orchestrator = orchestrator
        self.message_service = MessageService()
        self.recontextualizer = QueryRecontextualizer()

    async def load_ia_area_config(self, id_ia_area: int, db: Session) -> str:
        """
        Load IA area configuration from database using stored procedure.
        Falls back to LLM_ROLE_BEHAVIOR from env if SP returns no value or fails.

        Args:
            id_ia_area: ID of the IA area
            db: Database session

        Returns:
            Configuration string (max 1000 characters) from SP or LLM_ROLE_BEHAVIOR from env
        """
        try:
            if not db:
                logger.warning("Database session not available in RagService, using llm_role_behavior from env")
                return settings.llm_role_behavior

            query = text("""
                EXEC SP_IA_AREA_CONFIG_LOAD
                @ID_IA_AREA = :id_ia_area
            """)

            result = db.execute(query, {
                'id_ia_area': id_ia_area
            })

            config_data = result.fetchone()
            result.close()

            if not config_data:
                logger.info(f"No config found for id_ia_area={id_ia_area}, using llm_role_behavior from env")
                return settings.llm_role_behavior

            config_dict = dict(config_data._mapping) if hasattr(config_data, '_mapping') else dict(zip(result.keys(), config_data))

            # Get the first value from the result (config text)
            config_text = list(config_dict.values())[0] if config_dict else None

            if config_text and str(config_text).strip():
                return str(config_text)[:1000]

            # No valid config text, use env fallback
            logger.info(f"No valid config found for id_ia_area={id_ia_area}, using llm_role_behavior from env")
            return settings.llm_role_behavior

        except Exception as e:
            logger.error(f"Error loading IA area config for id_ia_area={id_ia_area}: {e}, using llm_role_behavior from env")
            return settings.llm_role_behavior

    @staticmethod
    def clean_user_query(message: str) -> str:
        """
        Clean user query by removing special quotes, normalizing whitespace, and handling line breaks.

        Args:
            message: Raw user input message

        Returns:
            Cleaned query string
        """
        # First strip leading/trailing whitespace and line breaks
        user_query = message.strip()

        # Replace newlines/line breaks in the middle with spaces
        user_query = user_query.replace('\n', ' ').replace('\r', ' ')

        # Remove special quote characters from anywhere in the string
        special_quotes = ['"', '“', '”', "'"]
        for quote in special_quotes:
            user_query = user_query.replace(quote, '')

        # Normalize multiple spaces into single space and trim again
        user_query = re.sub(r'\s+', ' ', user_query).strip()

        return user_query


    def _build_rag_prompt(self, message: str, context_text: str) -> str:
        """
        Build the RAG prompt with context and user message.
        Delegates to model-specific configuration for optimal prompts.
        """
        # Get model configuration from LLM provider
        model_config = self.llm_provider.get_model_config()

        # Use model-specific prompt building (conversation history handled by Converse API)
        return model_config.build_rag_prompt(message, context_text)


    async def agent_orchestrator_stream(self, user_id: int, user: str, message: str, company_id: int, company: str, area_id: int, area: str, id_ia_area: int, db: Session, top_k: int = None, similarity_threshold: float = None, alpha: float = None, temperature: float = None, max_tokens: int = None, external_token: str = None):
        """
        Analyze user query using the agent orchestrator model to determine workflow requirements.
        Enhanced version that accepts all process_rag_query_stream parameters for complete context.
        """
        try:
            # Validate inputs
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")
            if not user_id:
                raise ValueError("User ID is required")
            if not company or not company.strip():
                raise ValueError("Company ID is required and cannot be empty")
            if not area or not area.strip():
                raise ValueError("Area is required and cannot be empty")
            if id_ia_area is None:
                raise ValueError("ID IA Area is required and cannot be empty")
            if not external_token or not external_token.strip():
                raise ValueError("External token is required and cannot be empty")

            # Use injected orchestrator
            if not self.orchestrator:
                raise ValueError("Orchestrator not configured for this service instance")

            orchestrator = self.orchestrator

            # Load IA area role behavior configuration
            role_behavior = await self.load_ia_area_config(id_ia_area, db)

            # Clean user query
            user_query = self.clean_user_query(message)

            # Define available APIs (this could be loaded from config)
            available_apis = [
                {
                    "method": "GET",
                    "endpoint": "/bdt/talent/list",
                    "description": "Table: talents, Columns: idTalento, nombres, apellidoPaterno, apellidoMaterno, imagen, puesto, pais, ciudad, idModalidadFacturacion, montoInicialPlanilla, montoFinalPlanilla, montoInicialRxH, montoFinalRxH, moneda, estrellas, esFavorito, idMonedaPlan, idMonedaRxh",
                    "params": {
                        "nPag": { "type": "integer", "required": False },
                        "search": { "type": "string", "required": False },
                        "techAbilities": { "type": "string", "required": False },
                        "idEnglishLevel": { "type": "integer", "required": False },
                        "idTalentCollection": { "type": "integer", "required": False }
                    }
                }
            ]

            # Call orchestrator to analyze the query
            analysis = await orchestrator.analyze_query(user_query, available_apis)

            # Generate tasks from analysis
            tasks = TaskGenerator.generate_tasks_from_analysis(analysis, available_apis, user_query)

            # Execute tasks sequentially
            query_embedding = None
            context_text = ""
            execution_failed = False

            for task in tasks:
                if execution_failed:
                    break
                if task.get("action") == "embedding":
                    try:
                        query_text = task.get("input", user_query)
                        query_embedding = await self.generate_embedding(query_text)
                    except Exception as e:
                        execution_failed = True
                        break  # Stop execution if embedding fails

                elif task.get("action") == "retrieval":
                    try:
                        if query_embedding is None:
                            # Generate embedding if not already done
                            query_embedding = await self.generate_embedding(user_query)

                        # Use semantic_query from analysis if available, otherwise use user_query
                        semantic_query = analysis.get("semantic_query", "").strip() if analysis.get("semantic_query") else user_query

                        search_results = await self.search_by_embedding_hybrid(
                            query_text=semantic_query,
                            query_embedding=query_embedding,
                            company_id=company_id,
                            area_id=area_id,
                            top_k=top_k,
                            similarity_threshold=similarity_threshold,
                            alpha=alpha
                        )
                        # Build context from retrieved documents
                        if search_results["documents"]:
                            context_parts = []
                            for doc in search_results["documents"]:
                                if doc["content"]:
                                    # Formato de referencia de páginas
                                    if doc.get("page_start") is not None and doc.get("page_end") is not None:
                                        if doc["page_start"] == doc["page_end"]:
                                            page_ref = f"Page {doc['page_start']}"
                                        else:
                                            page_ref = f"Pages {doc['page_start']}-{doc['page_end']}"
                                    else:
                                        page_ref = "Page N/A"
                                    # Armar metadata extra
                                    source_info = []
                                    if doc.get("doc_id"): 
                                        source_info.append(f"ID: {doc['doc_id']}")
                                    if doc.get("doc_title"): 
                                        source_info.append(f"Title: {doc['doc_title']}")
                                    if doc.get("section_title"): 
                                        source_info.append(f"Section: {doc['section_title']}")
                                    if doc.get("section_path"): 
                                        source_info.append(f"Path: {doc['section_path']}")
                                    if doc.get("score"): 
                                        source_info.append(f"Score: {doc['score']}")
                                    # Construcción del bloque final
                                    joined_sources = '\n'.join(source_info)
                                    context_parts.append(
                                        f"Source: {joined_sources}, {page_ref}\nContent:\n{doc['content']}"
                                    )

                            context_text = "\n\n".join(context_parts)
                    except Exception as e:
                        execution_failed = True
                        break  # Stop execution if retrieval fails

                elif task.get("action") == "api_call":
                    try:
                        method = task.get("method", "GET").upper()
                        if method == "GET":
                            api_result = await httpx_get(task.get("endpoint", ""), external_token, task.get("params", {}))
                        else:
                            api_result = await httpx_post(task.get("endpoint", ""), external_token, task.get("params", {}))

                        # Add API result to context only if there's actual data
                        if api_result.get("success") and api_result.get("data"):
                            api_data = json.dumps(api_result['data'])
                            if api_data and api_data.strip() not in ["{}", "[]", "null"]:
                                # Format API call result with metadata
                                api_context_parts = []
                                api_context_parts.append(f"API Call:")
                                api_context_parts.append(f"Endpoint: {task.get('endpoint', 'N/A')}")
                                api_context_parts.append(f"Params: {json.dumps(task.get('params', {}))}")
                                api_context_parts.append(f"Response:\n{api_data}")

                                formatted_api_context = "\n".join(api_context_parts)

                                # Add separator if context already has content
                                if context_text:
                                    context_text += "\n\n"
                                context_text += formatted_api_context

                    except Exception as e:
                        execution_failed = True
                        break  # Stop execution if API call fails

                elif task.get("action") == "llm_response":
                    try:
                        # Check if we have any context at all
                        if not context_text or context_text.strip() == "":
                            # No context available - return predefined message
                            yield {
                                "type": "chunk",
                                "content": NO_CONTEXT_MESSAGE
                            }
                            break

                        # Prepare the query, checking for format requirements
                        query_to_use = user_query
                        task_format = task.get("format")
                        if task_format:
                            if task_format.lower() == "list":
                                query_to_use = f"{user_query}. IMPORTANT: Format the list items as a compact markdown list in a single line per item, including only the key and necessary information for clear understanding. If there is additional text content after the list, continue with it as plain text below the list."
                            elif task_format.lower() == "table":
                                query_to_use = f"{user_query}. IMPORTANT: Provide your response in the same language as the question. If explanation or summary is needed, include it briefly before the table. Then present the tabular data as a well-structured markdown table, showing ALL rows and ALL columns without adding empty or duplicate rows. Convert headers to natural language in the same language as the question, and ensure the table is properly aligned and easy to read. If there is additional text content after the table, continue with it as plain text below the table."

                        # Build prompt with context (we know context_text exists here)
                        prompt = self._build_rag_prompt(query_to_use, context_text)

                        # Use provided parameters or fall back to environment defaults
                        llm_temperature = temperature if temperature is not None else settings.llm_temperature
                        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

                        # Track if assistant metadata has been sent
                        agent_timestamp_sent = False

                        # Stream response directly from LLM provider (same as process_rag_query_stream)
                        async for chunk in self.llm_provider.generate_stream(prompt, max_tokens=llm_max_tokens, temperature=llm_temperature, role_behavior=role_behavior):
                            # Send agent metadata on first chunk
                            if not agent_timestamp_sent:
                                agent_timestamp = str(int(time.time() * 1000))
                                yield {
                                    "type": "assistant_metadata",
                                    "sender": 2,  # 2 = agent
                                    "created_at": agent_timestamp
                                }
                                agent_timestamp_sent = True

                            # Yield each chunk for streaming (same format as process_rag_query_stream)
                            yield {
                                "type": "chunk",
                                "content": chunk
                            }

                    except Exception as e:
                        execution_failed = True
                        break  # Stop execution if LLM response fails

            # Send completion signal
            yield {
                "type": "complete",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error in agent_orchestrator_stream: {e}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}"
            }


    async def process_rag_query_stream(self, user_id: int, user: str, message: str, company_id: int, company: str, area_id: int, area: str, id_ia_area: int, db: Session, created_at: str, chat_id: str = None, top_k: int = None, similarity_threshold: float = None, alpha: float = None, temperature: float = None, max_tokens: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Proceso RAG completo con streaming: embeddings → search → LLM streaming → response
        """
        try:
            # Validate inputs
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")
            if not user_id:
                raise ValueError("User ID is required")
            if not company or not company.strip():
                raise ValueError("Company ID is required and cannot be empty")
            if not area or not area.strip():
                raise ValueError("Area is required and cannot be empty")
            if id_ia_area is None:
                raise ValueError("ID IA Area is required and cannot be empty")

        except ValueError as e:
            logger.error(f"Validation error in process_rag_query_stream: {e}")
            raise
        except Exception as e:
            logger.error(f"Initialization error in process_rag_query_stream: {e}")
            raise ConnectionError(f"RAG streaming service initialization failed: {str(e)}")

        # Get last 6 messages from chat history if chat_id exists
        conversation_analysis = []
        if chat_id is not None:
            try:
                messages_response = await self.message_service.get_last_n_messages(
                    chat_id=f"chat-{chat_id}",
                    n=6
                )
                # Format messages for context counter
                conversation_analysis = [
                    {
                        "role": "user" if msg.sender == 0 else "assistant",
                        "content": msg.message
                    }
                    for msg in messages_response
                ]
                # Reverse the list so messages are in correct chronological order (oldest first)
                conversation_analysis.reverse()
                logger.info(f"Retrieved {len(conversation_analysis)} messages from chat history for context")
            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {e}, continuing without history")
                conversation_analysis = []

        cleaned_message = self.clean_user_query(message)

        # Recontextualize query if chat_id exists
        recontextualized_result = None
        if chat_id is not None:
            try:
                recontextualized_result = await self.recontextualizer.recontextualize_query(
                    user_query=cleaned_message,
                    conversation_history=conversation_analysis
                )
                logger.info(f"Recontextualization result: {recontextualized_result}")
            except Exception as e:
                logger.warning(f"Failed to recontextualize query: {e}, continuing without recontextualization")
                recontextualized_result = None

        # Load IA area role behavior configuration
        role_behavior = await self.load_ia_area_config(id_ia_area, db)

        # Track if a new chat was created and store the title
        new_chat_created = False
        new_chat_titulo = None
        new_chat_timestamp = None

        # Create chat if chat_id is not provided
        if chat_id is None:
            from datetime import datetime
            now = datetime.now()
            formatted_date = now.strftime("%d/%m/%Y %H:%M")
            titulo = f"Nueva conversación {formatted_date}"

            # Create ChatRepository instance for this request
            chat_repository = ChatRepository(db)

            # Call stored procedure to create chat via repository
            new_chat_id = chat_repository.create_chat(
                id_usuario=user_id,
                id_area=area_id,
                id_empresa=company_id,
                titulo=titulo
            )

            if new_chat_id:
                chat_id = new_chat_id
                new_chat_created = True
                new_chat_titulo = titulo
                new_chat_timestamp = now.isoformat()
            else:
                # Chat creation failed - stop execution
                logger.error("Chat creation failed - no ID returned, stopping execution")
                yield {
                    "type": "error",
                    "message": "Failed to create chat",
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El chat no pudo crearse correctamente"
                    }
                }
                return

        # Save user message to DynamoDB
        if chat_id:
            try:
                await self.message_service.create_message(
                    chat_id=chat_id,
                    created_at=created_at,
                    sender=0,
                    message=cleaned_message
                )
            except Exception as e:
                logger.error(f"Failed to save user message: {e}")
                yield {
                    "type": "error",
                    "message": "Failed to save user message",
                    "result": {
                        "idTipoMensaje": 1,
                        "mensaje": "El mensaje del usuario no pudo guardarse correctamente"
                    }
                }
                return

        # Yield metadata early (before processing) so frontend can show "Pensando..." placeholder
        yield {
            "type": "metadata",
            "llm_model_used": settings.llm_model_id,
            "chat_id": chat_id
        }

        # If a new chat was created, send the chat object to the frontend
        if new_chat_created:
            yield {
                "type": "chat_created",
                "chat": {
                    "ID_CHAT": chat_id,
                    "ID_AREA": area_id,
                    "ID_EMPRESA": company_id,
                    "TITULO": new_chat_titulo,
                    "ULTIMO_MENSAJE_FECHA": new_chat_timestamp,
                    "ID_ESTADO_REGISTRO": 1
                }
            }

        # Step 1: Determine query to use for embedding and search
        query_for_search = cleaned_message
        if recontextualized_result and recontextualized_result.get("needs_context", False):
            recontextualized_query = recontextualized_result.get("response", "").strip()
            if recontextualized_query:
                query_for_search = recontextualized_query
                logger.info(f"Using recontextualized query for search: {query_for_search}")
            else:
                logger.warning("Recontextualized response is empty, using original cleaned message")
        else:
            logger.info(f"Using original query for search (needs_context={recontextualized_result.get('needs_context', 'N/A') if recontextualized_result else 'N/A'})")

        # Generate embedding for the query (either original or recontextualized)
        query_embedding = await self.generate_embedding(query_for_search)

        # Step 2: Search vector database using the embedding (hybrid search)
        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        search_result = await self.search_by_embedding_hybrid(
            query_text=query_for_search,
            query_embedding=query_embedding,
            company_id=company_id,
            area_id=area_id,
            top_k=search_top_k,
            similarity_threshold=search_threshold,
            alpha=alpha
        )

        # Add query to result for compatibility
        search_result["query"] = cleaned_message

        # Step 3: Prepare context text for LLM with source metadata
        context_with_sources = []

        # Only process documents if any were found
        if search_result["total_found"] > 0:
            for doc in search_result["documents"]:
                if doc["content"]:
                    # Formato de referencia de páginas
                    if doc.get("page_start") is not None and doc.get("page_end") is not None:
                        if doc["page_start"] == doc["page_end"]:
                            page_ref = f"Page {doc['page_start']}"
                        else:
                            page_ref = f"Pages {doc['page_start']}-{doc['page_end']}"
                    else:
                        page_ref = "Page N/A"
                    # Armar metadata extra
                    source_info = []
                    if doc.get("doc_id"):
                        source_info.append(f"ID: {doc['doc_id']}")
                    if doc.get("doc_title"):
                        source_info.append(f"Title: {doc['doc_title']}")
                    if doc.get("section_title"):
                        source_info.append(f"Section: {doc['section_title']}")
                    if doc.get("section_path"):
                        source_info.append(f"Path: {doc['section_path']}")
                    if doc.get("score"):
                        source_info.append(f"Score: {doc['score']}")
                    # Construcción del bloque final
                    joined_sources = '\n'.join(source_info)
                    context_with_sources.append(
                        f"Source: {joined_sources}, {page_ref}\nContent:\n{doc['content']}"
                    )

        # Build context text (will be empty string if no documents found)
        context_text = "\n\n".join(context_with_sources) if context_with_sources else ""

        # Step 4: Fetch additional conversation history for LLM prompt if needed
        # Based on recontextualization result flags
        conversation_history_for_prompt = []
        if chat_id and recontextualized_result:
            needs_context = recontextualized_result.get("needs_context", False)
            summary_intent = recontextualized_result.get("summary_intent", False)

            # Determine how many messages to fetch based on flags
            messages_to_fetch = 0
            if needs_context and summary_intent:
                # Both flags true: fetch 16 messages
                messages_to_fetch = 16
                logger.info("Fetching 16 messages for LLM prompt (needs_context=true, summary_intent=true)")
            elif needs_context:
                # Only needs_context true: fetch 8 messages
                messages_to_fetch = 8
                logger.info("Fetching 8 messages for LLM prompt (needs_context=true)")
            elif summary_intent:
                # Only summary_intent true: fetch 16 messages
                messages_to_fetch = 16
                logger.info("Fetching 16 messages for LLM prompt (summary_intent=true)")
            # If both false: don't fetch any messages (messages_to_fetch = 0)

            # Fetch messages if needed
            if messages_to_fetch > 0:
                try:
                    history_messages = await self.message_service.get_last_n_messages(
                        chat_id=f"chat-{chat_id}",
                        n=messages_to_fetch
                    )
                    conversation_history_for_prompt = [
                        {
                            "role": "user" if msg.sender == 0 else "assistant",
                            "content": msg.message
                        }
                        for msg in history_messages
                    ]
                    # Reverse the list so messages are in correct chronological order (oldest first)
                    conversation_history_for_prompt.reverse()
                    logger.info(f"Retrieved {len(conversation_history_for_prompt)} messages for LLM prompt")
                except Exception as e:
                    logger.warning(f"Failed to retrieve conversation history for prompt: {e}")
                    conversation_history_for_prompt = []

        # Step 5: Generate LLM answer
        # Build RAG prompt with context (conversation history handled by Converse API messages)
        rag_prompt = self._build_rag_prompt(cleaned_message, context_text)

        # Step 4: Generate streaming response using LLM
        # Use provided parameters or fall back to environment defaults
        llm_temperature = temperature if temperature is not None else settings.llm_temperature
        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        # Send assistant_metadata BEFORE starting LLM streaming
        # This gives frontend time to render the empty "Pensando..." placeholder
        assistant_timestamp = str(int(time.time() * 1000))
        yield {
            "type": "assistant_metadata",
            "sender": 1,
            "created_at": assistant_timestamp
        }

        # Accumulate assistant response chunks
        assistant_response = ""
        first_chunk_sent = False

        # Stream the LLM response with role behavior and conversation history
        async for chunk in self.generate_text_stream(
            prompt=rag_prompt,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
            role_behavior=role_behavior,
            messages=conversation_history_for_prompt if conversation_history_for_prompt else None
        ):
            assistant_response += chunk

            # Update chat last message date when first chunk with content is sent
            if not first_chunk_sent and chunk.strip():
                # Create ChatRepository instance for this update
                chat_repository = ChatRepository(db)
                chat_repository.update_ultimo_mensaje_fecha(chat_id)
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

    async def generate_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query text.
        """
        try:
            query_embedding = await self.embeddings_provider.embed(query)
            return query_embedding
        except ConnectionError as e:
            logger.error(f"Connection error during embedding generation: {e}")
            raise ConnectionError(f"Embedding service unavailable: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input for embedding: {e}")
            raise ValueError(f"Invalid query for embedding: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            raise ConnectionError(f"Embedding generation failed: {str(e)}")

    async def search_by_embedding_hybrid(self, query_text: str, query_embedding: List[float], company_id: int, area_id: int, top_k: int = None, similarity_threshold: float = None, alpha: float = None) -> Dict[str, Any]:
        """
        Hybrid search (vector + BM25) using pre-generated embedding and query text.
        Company ID and Area ID are concatenated with prefixes to form collection name and area filter.
        """
        from app.core.config import settings

        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        # Concatenate IDs with prefixes to form collection name and area filter
        company = f"EMPR{company_id}"
        area = f"AREA{area_id}"

        try:
            # Hybrid search (vector + BM25) - company is the collection name
            search_results = await self.vectorstore.search_in_collection_hybrid(
                collection_name=company,
                query_text=query_text,
                query_vector=query_embedding,
                area=area,
                top_k=search_top_k,
                similarity_threshold=search_threshold,
                alpha=alpha
            )
        except ConnectionError as e:
            logger.error(f"Connection error during hybrid search: {e}")
            raise ConnectionError(f"Vector database unavailable: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid search parameters: {e}")
            raise ValueError(f"Invalid search parameters: {str(e)}")
        except TimeoutError as e:
            logger.error(f"Timeout error during hybrid search: {e}")
            raise TimeoutError(f"Hybrid search timeout: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during hybrid search: {e}")
            raise ConnectionError(f"Hybrid search failed: {str(e)}")

        # Format results
        documents = []
        for result in search_results:
            metadata = result.get("metadata", {})
            documents.append({
                "content": result.get("content", ""),
                # Document metadata
                "doc_id": metadata.get("doc_id", ""),
                "doc_title": metadata.get("doc_title", ""),
                "section_title": metadata.get("section_title", ""),
                "section_path": metadata.get("section_path", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "token_count": metadata.get("token_count"),
                # Search metadata (hybrid returns score, not distance)
                "score": metadata.get("score"),
                "relevance_score": metadata.get("relevance_score"),
                "search_type": metadata.get("search_type", "hybrid")
            })

        return {
            "documents": documents,
            "total_found": len(documents),
            "search_parameters": {
                "top_k": search_top_k,
                "similarity_threshold": search_threshold,
                "alpha": alpha if alpha is not None else settings.rag_hybrid_alpha,
                "embedding_model": settings.embeddings_model_id,
                "search_type": "hybrid"
            },
            "embedding_dimensions": len(query_embedding),
            "status": "success"
        }

    async def generate_text_stream(
        self,
        prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        role_behavior: str = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text response using LLM.

        Args:
            prompt: The user prompt (used if messages is None)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            role_behavior: Optional role behavior (system prompt)
            messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                     If provided, prompt will be ignored and messages will be used instead
        """
        from app.core.config import settings

        # Validate that either prompt or messages is provided
        if messages is None and (not prompt or not prompt.strip()):
            raise ValueError("Either prompt or messages must be provided")

        # Use provided values or fall back to config defaults
        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        llm_temperature = temperature if temperature is not None else settings.llm_temperature

        # Validate parameters
        if llm_max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if not (0.0 <= llm_temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        try:
            has_content = False

            async for chunk in self.llm_provider.generate_stream(
                prompt=prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                role_behavior=role_behavior,
                messages=messages
            ):
                # Detect stop reason signal
                if chunk.startswith("__STOP_REASON__:"):
                    stop_reason = chunk.split(":")[1]
                    if stop_reason == "max_tokens":
                        # Yield the error as a regular message instead of throwing exception
                        yield "⚠️ El modelo agotó los tokens disponibles durante el análisis de la consulta. Por favor, intenta con una pregunta más específica o reduce la complejidad de tu solicitud."
                        has_content = True
                    continue

                has_content = True
                yield chunk

            if not has_content:
                raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        except ConnectionError as e:
            logger.error(f"Connection error during LLM generation: {e}")
            raise ConnectionError(f"LLM service unavailable: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input for LLM: {e}")
            raise ValueError(f"Invalid prompt or parameters: {str(e)}")
        except TimeoutError as e:
            logger.error(f"Timeout error during LLM generation: {e}")
            raise TimeoutError(f"LLM generation timeout: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {e}")
            raise ConnectionError(f"LLM generation failed: {str(e)}")

