from typing import Tuple, List, Dict, Any, AsyncGenerator
import logging
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort
from app.domain.ports.llm_port import LLMPort
from app.core.config import settings

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

    def __init__(self, embeddings_provider: EmbeddingsPort, vectorstore: VectorStorePort, llm_provider: LLMPort):
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
        self.llm_provider = llm_provider
    
    def _build_rag_prompt(self, message: str, context_text: str) -> str:
        """
        Build the RAG prompt with context and user message.
        Uses different prompt styles optimized for different model types.
        """
        from app.core.config import settings
        
        # Check if using Claude model
        model_id_lower = settings.llm_model_id.lower()
        is_claude = "claude" in model_id_lower or "anthropic" in model_id_lower
        
        if is_claude:
            # Claude-optimized prompt: concise, natural, conversational
            return f"""Based on the following context, please answer the user's question. Include relevant source references with document ID and page numbers. Do not search on internet. Answer in the same language as the question.

Context:
{context_text}

Question: {message}

Please provide a clear, accurate response based solely on the provided context."""
        else:
            # Llama-optimized prompt: explicit instructions, structured format
            return f"""Answer directly and include the source references. Use ONLY the context provided to answer. Do NOT explain, justify, or comment on the correctness. Do NOT repeat text. Always cite your sources using the document ID and page numbers provided in the context.

{context_text}

Q: {message}
A:"""



    async def process_rag_query_stream(self, user_id: str, message: str, company_id: str, area: str = None, collection: str = None, top_k: int = None, similarity_threshold: float = None, temperature: float = None, max_tokens: int = None) -> AsyncGenerator[Dict[str, Any], None]:
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
        
        # Step 1: Generate embedding for the query
        query_embedding = await self.generate_embedding(message)

        # Step 2: Search vector database using the embedding
        # Use provided parameters or fall back to environment defaults
        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        search_result = await self.search_by_embedding(
            query_embedding=query_embedding,
            company_id=company_id,
            area=area,
            collection=collection,
            top_k=search_top_k,
            similarity_threshold=search_threshold
        )

        # Add query to result for compatibility
        search_result["query"] = message
        
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
                # Format page reference more accurately
                if doc['page_start'] == doc['page_end']:
                    page_ref = f"Page {doc['page_start']}"
                else:
                    page_ref = f"Pages {doc['page_start']}-{doc['page_end']}"
                source_info = f"[Source: Document {doc['doc_id']}, {page_ref}]"
                context_with_sources.append(f"{doc['content']}\n{source_info}")
        
        context_text = "\n\n".join(context_with_sources)
        
        # Step 4: Generate LLM answer
        
        # Normal RAG flow with context
        rag_prompt = self._build_rag_prompt(message, context_text)
        
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

        Args:
            query: Text to convert to embedding

        Returns:
            List of float values representing the embedding
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

    async def search_by_embedding(self, query_embedding: List[float], company_id: str = None, area: str = None, collection: str = None, top_k: int = None, similarity_threshold: float = None) -> Dict[str, Any]:
        """
        Search vector database using pre-generated embedding.

        Args:
            query_embedding: Pre-generated embedding vector
            company_id: Company identifier for filtering results
            area: Area identifier for additional filtering. When specified, returns documents
                  from both the specific area AND the "Default" area (which are visible to all areas)
            collection: Collection/class name to search in (defaults to config or company_id)
            top_k: Number of results to return (defaults to config)
            similarity_threshold: Minimum similarity score (defaults to config)

        Returns:
            Dict with search results and metadata
        """
        from app.core.config import settings

        # Use provided values or fall back to config defaults
        # If company_id is provided and no collection specified, use company_id as collection name
        if collection is not None:
            search_collection = collection
        elif company_id is not None:
            search_collection = company_id  # Use company_id as collection name
        else:
            search_collection = settings.weaviate_class_name

        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

        try:
            # Search vector database in specified collection with company and area filtering
            search_results = await self.vectorstore.search_in_collection(
                collection_name=search_collection,
                query_vector=query_embedding,
                top_k=search_top_k,
                similarity_threshold=search_threshold,
                company_id=company_id,
                area=area
            )
        except ConnectionError as e:
            logger.error(f"Connection error during vector search: {e}")
            raise ConnectionError(f"Vector database unavailable: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid search parameters: {e}")
            raise ValueError(f"Invalid search parameters: {str(e)}")
        except TimeoutError as e:
            logger.error(f"Timeout error during vector search: {e}")
            raise TimeoutError(f"Vector search timeout: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during vector search: {e}")
            raise ConnectionError(f"Vector search failed: {str(e)}")

        # Format results
        documents = []
        for result in search_results:
            metadata = result.get("metadata", {})
            documents.append({
                "content": result.get("content", ""),
                # All database parameters (matching CargaConocimiento_iA schema)
                "company_id": metadata.get("company_id", ""),
                "doc_id": metadata.get("doc_id", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "token_count": metadata.get("token_count"),
                # Search metadata
                "distance": metadata.get("distance", 0.0),
                "relevance_score": metadata.get("relevance_score", 1.0 - metadata.get("distance", 0.0))
            })

        return {
            "documents": documents,
            "total_found": len(documents),
            "search_parameters": {
                "top_k": search_top_k,
                "similarity_threshold": search_threshold,
                "embedding_model": settings.embeddings_model_id,
                "collection": search_collection,
                "company_id": company_id,
                "area": area
            },
            "embedding_dimensions": len(query_embedding),
            "status": "success"
        }

    async def generate_text_stream(self, prompt: str, max_tokens: int = None, temperature: float = None) -> AsyncGenerator[str, None]:
        """
        Generate streaming text response using LLM.

        Args:
            prompt: Text prompt to send to LLM
            max_tokens: Maximum tokens to generate (defaults to config)
            temperature: Sampling temperature (defaults to config)

        Yields:
            Text chunks as they are generated by the LLM
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


