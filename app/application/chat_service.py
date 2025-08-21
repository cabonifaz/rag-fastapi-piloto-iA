from typing import Tuple, List, Dict, Any
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
            "model_id": model_id,
            "embedding_dimensions": len(embedding),
            "embedding": embedding,
            "status": "success"
        }

    async def process_rag_query(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Complete RAG flow: embedding generation + vector search + context assembly.
        
        Args:
            user_id: User identifier for potential session management
            message: User's question/message
            
        Returns:
            Dict with answer, context documents, and metadata
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Use get_chat_service_with_vectorstore() for RAG functionality.")
        
        # Step 1: Generate embedding for the user's message
        query_embedding = await self.embeddings_provider.embed(message)
        
        # Step 2: Search for relevant documents in the vector database
        search_results = await self.vectorstore.search(
            query_vector=query_embedding,
            top_k=5,
            similarity_threshold=0.7
        )
        
        # Step 3: Extract context from search results
        context_documents = []
        for result in search_results:
            context_documents.append({
                "content": result.get("content", ""),
                "source": result.get("metadata", {}).get("source", ""),
                "distance": result.get("metadata", {}).get("distance", 0.0)
            })
        
        # Step 4: Prepare context text for potential LLM usage
        context_text = "\n\n".join([doc["content"] for doc in context_documents if doc["content"]])
        
        return {
            "user_id": user_id,
            "message": message,
            "context_documents": context_documents,
            "context_text": context_text,
            "total_documents_found": len(context_documents),
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
