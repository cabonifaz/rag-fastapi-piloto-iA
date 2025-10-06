from typing import Tuple, List, Dict, Any, AsyncGenerator
import logging
import json
import re
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.task_decomposition_port import QueryAnalysisPort
from app.core.config import settings
from app.infrastructure.task_decomposition.task_generator import TaskGenerator
from app.infrastructure.api_clients.api_client import httpx_get, httpx_post

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NO_CONTEXT_MESSAGE = "Parece que tu pregunta no es lo suficientemente específica 🤔. ¿Me das un poco más de contexto para ayudarte mejor?"


class ChatService:
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

        # Use model-specific prompt building
        return model_config.build_rag_prompt(message, context_text)


    async def agent_orchestrator_stream(self, user_id: str, message: str, company_id: str, area: str = None, top_k: int = None, similarity_threshold: float = None, alpha: float = None, temperature: float = None, max_tokens: int = None, external_token: str = None):
        """
        Analyze user query using the agent orchestrator model to determine workflow requirements.
        Enhanced version that accepts all process_rag_query_stream parameters for complete context.
        """
        try:
            # Use injected orchestrator
            if not self.orchestrator:
                raise ValueError("Orchestrator not configured for this service instance")

            orchestrator = self.orchestrator

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
                            area=area,
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
                                    context_parts.append(
                                        f"Source: {'\n'.join(source_info)}, {page_ref}\nContent:\n{doc['content']}"
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

                        # Stream response directly from LLM provider (same as process_rag_query_stream)
                        async for chunk in self.llm_provider.generate_stream(prompt, max_tokens=llm_max_tokens, temperature=llm_temperature):
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


    async def process_rag_query_stream(self, user_id: str, message: str, company_id: str, area: str = None, top_k: int = None, similarity_threshold: float = None, alpha: float = None, temperature: float = None, max_tokens: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Proceso RAG completo con streaming: embeddings → search → LLM streaming → response
        """
        try:
            # Validate inputs
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")
            if not user_id:
                raise ValueError("User ID is required")
            if not company_id or not company_id.strip():
                raise ValueError("Company ID is required and cannot be empty")
            if not area or not area.strip():
                raise ValueError("Area is required and cannot be empty")
                
        except ValueError as e:
            logger.error(f"Validation error in process_rag_query_stream: {e}")
            raise
        except Exception as e:
            logger.error(f"Initialization error in process_rag_query_stream: {e}")
            raise ConnectionError(f"RAG streaming service initialization failed: {str(e)}")

        # Clean the user message
        cleaned_message = self.clean_user_query(message)

        # Step 1: Generate embedding for the query
        query_embedding = await self.generate_embedding(cleaned_message)

        # Step 2: Search vector database using the embedding (hybrid search)
        # Use provided parameters or fall back to environment defaults
        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        search_result = await self.search_by_embedding_hybrid(
            query_text=cleaned_message,
            query_embedding=query_embedding,
            company_id=company_id,
            area=area,
            top_k=search_top_k,
            similarity_threshold=search_threshold,
            alpha=alpha
        )

        # Add query to result for compatibility
        search_result["query"] = cleaned_message
        
        # Check if no documents found at database level
        if search_result["total_found"] == 0:
            # No documents found - return predefined message without LLM call
            yield {
                "type": "metadata",
                "user_id": user_id,
                "message": message,
                "llm_model_used": None,
            }
            yield {
                "type": "chunk",
                "content": NO_CONTEXT_MESSAGE
            }
            yield {
                "type": "complete",
                "status": "success"
            }
            return
        
        # Step 3: Prepare context text for LLM with source metadata
        context_with_sources = []
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
                context_with_sources.append(
                    f"Source: {'\n'.join(source_info)}, {page_ref}\nContent:\n{doc['content']}"
                )
        
        context_text = "\n\n".join(context_with_sources)
        
        # Step 4: Generate LLM answer
        
        # Normal RAG flow with context
        rag_prompt = self._build_rag_prompt(cleaned_message, context_text)
        
        # Step 4: Generate streaming response using LLM
        # Use provided parameters or fall back to environment defaults
        llm_temperature = temperature if temperature is not None else settings.llm_temperature
        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        
        # Yield metadata first (matching /chat response format - no context exposed)
        yield {
            "type": "metadata",
            "user_id": user_id,
            "message": message,
            "llm_model_used": settings.llm_model_id,
        }
        
        # Stream the LLM response
        async for chunk in self.generate_text_stream(rag_prompt, max_tokens=llm_max_tokens, temperature=llm_temperature):
            yield {
                "type": "chunk",
                "content": chunk
            }
        
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

    async def search_by_embedding_hybrid(self, query_text: str, query_embedding: List[float], company_id: str, area: str, top_k: int = None, similarity_threshold: float = None, alpha: float = None) -> Dict[str, Any]:
        """
        Hybrid search (vector + BM25) using pre-generated embedding and query text.
        """
        from app.core.config import settings

        # Use company_id as collection name, fallback to default
        if company_id is not None:
            search_collection = company_id
        else:
            search_collection = settings.weaviate_class_name

        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        try:
            # Hybrid search (vector + BM25) in specified collection with company and area filtering
            search_results = await self.vectorstore.search_in_collection_hybrid(
                collection_name=search_collection,
                query_text=query_text,
                query_vector=query_embedding,
                company_id=company_id,
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

    async def generate_text_stream(self, prompt: str, max_tokens: int = None, temperature: float = None) -> AsyncGenerator[str, None]:
        """
        Generate streaming text response using LLM.
        """
        from app.core.config import settings

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use provided values or fall back to config defaults
        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        llm_temperature = temperature if temperature is not None else settings.llm_temperature

        # Validate parameters
        if llm_max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if not (0.0 <= llm_temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        try:
            async for chunk in self.llm_provider.generate_stream(prompt, max_tokens=llm_max_tokens, temperature=llm_temperature):
                yield chunk

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

