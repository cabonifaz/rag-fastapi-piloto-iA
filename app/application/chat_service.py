from typing import Tuple, List, Dict, Any, AsyncGenerator
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vectorstore_port import VectorStorePort


class ChatService:
    """
    Application service that orchestrates RAG flow following hexagonal architecture:
    1. Generates embeddings using the embeddings port
    2. Searches context in vector database using vectorstore port
    3. Calls LLM with history + context
    4. Returns response
    """

    def __init__(self, embeddings_provider: EmbeddingsPort, vectorstore: VectorStorePort):
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
    
    def _build_rag_prompt(self, message: str, context_text: str) -> str:
        """
        Build the RAG prompt with context and user message.
        Centralized prompt template for both regular and streaming endpoints.
        """
        return f"""Answer directly. Use ONLY the context provided to answer. Do NOT repeat text. If you already answer the question, STOP.

{context_text}

Q: {message}
A:"""

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text.
        This method encapsulates the embedding generation logic.
        """
        return await self.embeddings_provider.embed(text)

    async def test_embedding(self, text: str, model_id: str) -> Dict[str, Any]:
        """
        Test embedding generation with detailed response.
        Returns both the embedding and metadata for testing purposes.
        """
        embedding = await self.embeddings_provider.embed(text)
        
        return {
            "message": text,
            "embedding_model": model_id,
            "embedding_dimensions": len(embedding),
            "embedding": embedding,
            "status": "success"
        }

    async def process_rag_query(self, user_id: str, message: str, company_id: str, collection: str = None, top_k: int = None, similarity_threshold: float = None, temperature: float = None, max_tokens: int = None, llm_provider=None) -> Dict[str, Any]:
        """
        Complete RAG flow: embedding generation + vector search + context assembly + LLM answer generation.
        
        Args:
            user_id: User identifier for potential session management
            message: User's question/message
            llm_provider: Optional LLM provider for answer generation
            
        Returns:
            Dict with answer, context documents, and metadata
        """
        from app.core.config import settings
        
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Use get_chat_service_with_vectorstore() for RAG functionality.")
        
        # Step 1: Use the search_documents method for consistent vector search
        # Use provided parameters or fall back to environment defaults
        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold
        
        search_result = await self.search_documents(
            query=message,
            company_id=company_id,
            collection=collection,
            top_k=search_top_k,
            similarity_threshold=search_threshold
        )
        
        # Check if no documents found at database level
        if search_result["total_found"] == 0:
            # No documents found - return predefined message without LLM call
            return {
                "user_id": user_id,
                "message": message,
                "answer": "There is no information available about this subject in the database.",
                "context_documents": [],
                "context_text": "",
                "total_documents_found": 0,
                "embedding_dimensions": search_result["embedding_dimensions"],
                "collection_searched": settings.weaviate_class_name,
                "llm_model_used": None,
                "search_parameters": search_result["search_parameters"],
                "status": "success"
            }
        
        # Step 2: Extract documents from search result
        context_documents = []
        for doc in search_result["documents"]:
            context_documents.append({
                "content": doc["content"],
                "company_id": doc["company_id"],
                "doc_id": doc["doc_id"],
                "chunk_id": doc["chunk_id"],
                "page_start": doc["page_start"],
                "page_end": doc["page_end"],
                "char_start": doc["char_start"],
                "char_end": doc["char_end"],
                "token_count": doc["token_count"],
                "distance": doc["distance"],
                "relevance_score": doc["relevance_score"]
            })
        
        # Step 3: Prepare context text for LLM
        context_text = "\n\n".join([doc["content"] for doc in context_documents if doc["content"]])
        
        # Step 4: Generate LLM answer (LLM is required for /chat endpoint)
        if not llm_provider:
            raise ValueError("LLM provider is required for /chat endpoint but was not provided")
        
        # Normal RAG flow with context
        rag_prompt = self._build_rag_prompt(message, context_text)
        
        # Use provided parameters or fall back to environment defaults
        llm_temperature = temperature if temperature is not None else settings.llm_temperature
        llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        
        answer = await llm_provider.generate(rag_prompt, max_tokens=llm_max_tokens, temperature=llm_temperature)
        llm_model_used = settings.llm_model_id
        
        # Step 5: Return complete RAG response
        return {
            "user_id": user_id,
            "message": message,
            "answer": answer,
            "context_documents": context_documents,
            "context_text": context_text,
            "total_documents_found": len(context_documents),
            "embedding_dimensions": search_result["embedding_dimensions"],
            "collection_searched": settings.weaviate_class_name,
            "llm_model_used": llm_model_used,
            "search_parameters": search_result["search_parameters"],
            "status": "success"
        }

    async def process_rag_query_stream(self, user_id: str, message: str, company_id: str, collection: str = None, top_k: int = None, similarity_threshold: float = None, temperature: float = None, max_tokens: int = None, llm_provider=None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Proceso RAG completo con streaming: embeddings → search → LLM streaming → response
        """
        from app.core.config import settings
        
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Use get_chat_service_with_vectorstore() for RAG functionality.")
        
        # Step 1: Use the search_documents method for consistent vector search
        # Use provided parameters or fall back to environment defaults
        search_top_k = top_k if top_k is not None else settings.rag_top_k_results
        search_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold
        
        search_result = await self.search_documents(
            query=message,
            company_id=company_id,
            collection=collection,
            top_k=search_top_k,
            similarity_threshold=search_threshold
        )
        
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
                "content": "There is no information available about this subject in the database."
            }
            yield {
                "type": "complete",
                "status": "success"
            }
            return
        
        # Step 2: Extract documents from search result
        context_documents = []
        for doc in search_result["documents"]:
            context_documents.append({
                "content": doc["content"],
                "company_id": doc["company_id"],
                "doc_id": doc["doc_id"],
                "chunk_id": doc["chunk_id"],
                "page_start": doc["page_start"],
                "page_end": doc["page_end"],
                "char_start": doc["char_start"],
                "char_end": doc["char_end"],
                "token_count": doc["token_count"],
                "distance": doc["distance"],
                "relevance_score": doc["relevance_score"]
            })
        
        # Step 3: Prepare context text for LLM
        context_text = "\n\n".join([doc["content"] for doc in context_documents if doc["content"]])
        
        # Step 4: Generate LLM answer (LLM is required for /chat endpoint)
        if not llm_provider:
            raise ValueError("LLM provider is required for /chat-streaming endpoint but was not provided")
        
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
        async for chunk in llm_provider.generate_stream(rag_prompt, max_tokens=llm_max_tokens, temperature=llm_temperature):
            yield {
                "type": "chunk",
                "content": chunk
            }
        
        # Final completion signal
        yield {
            "type": "complete",
            "status": "success"
        }

    async def search_documents(self, query: str, company_id: str = None, collection: str = None, top_k: int = None, similarity_threshold: float = None) -> Dict[str, Any]:
        """
        Basic vector database search implementation.
        
        Args:
            query: Search query text
            company_id: Company identifier for filtering results
            collection: Collection/class name to search in (defaults to config or company_id)
            top_k: Number of results to return (defaults to config)
            similarity_threshold: Minimum similarity score (defaults to config)
            
        Returns:
            Dict with search results and metadata
        """
        from app.core.config import settings
        
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Use get_chat_service_with_vectorstore() for search functionality.")
        
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
        
        # Step 1: Convert query to embedding
        query_embedding = await self.embeddings_provider.embed(query)
        
        # Step 2: Search vector database in specified collection with company filtering (no error handling)
        search_results = await self.vectorstore.search_in_collection(
            collection_name=search_collection,
            query_vector=query_embedding,
            top_k=search_top_k,
            similarity_threshold=search_threshold,
            company_id=company_id
        )
        
        # Step 3: Format results
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
            "query": query,
            "documents": documents,
            "total_found": len(documents),
            "search_parameters": {
                "top_k": search_top_k,
                "similarity_threshold": search_threshold,
                "embedding_model": settings.embeddings_model_id,
                "collection": search_collection,
                "company_id": company_id
            },
            "embedding_dimensions": len(query_embedding),
            "status": "success"
        }

    async def handle_message(self, session_id: str, user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
        self.memory_repository.save_message(session_id, role="user", content=user_message)

        query_vector = await self.embeddings_provider.embed(user_message)

        results = await self.vector_repository.search_by_vector(
            class_name="Documents",
            vector=query_vector,
            top_k=5,
            return_properties=["text", "source"],
        )

        context_chunks = [r["properties"].get("text", "") for r in results]

        history = self.memory_repository.get_history(session_id)
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        context_text = "\n".join(context_chunks)

        prompt = f"""
        You are an assistant. Use the following context to answer the question.
        
        Context:
        {context_text}

        Conversation so far:
        {formatted_history}

        User: {user_message}
        Assistant:
        """

        answer = await self.llm_provider.generate(prompt)

        self.memory_repository.save_message(session_id, role="assistant", content=answer)

        history = self.memory_repository.get_history(session_id)
        return answer, history
