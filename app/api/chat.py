from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from app.application.chat_service import ChatService
from app.core.config import settings
from app.core.container import container


router = APIRouter()


# Request Schema for endpoints that need company-specific search
class UnifiedRequest(BaseModel):
    user_id: str
    message: str
    company_id: str                    # Required, for company-specific search
    collection: str = None              # Optional, defaults to env config
    top_k: int = 5                     # Optional, for search endpoints
    similarity_threshold: float = 0.7   # Optional, for search endpoints
    temperature: Optional[float] = None  # Optional, defaults to env config
    max_tokens: Optional[int] = None     # Optional, defaults to env config

# Request Schema for embedding-only endpoints
class EmbeddingTestRequest(BaseModel):
    user_id: str
    message: str


# Response Models
class ChatResponse(BaseModel):
    user_id: str
    message: str
    answer: str
    llm_model_used: Optional[str] = None
    status: str


class EmbeddingTestResponse(BaseModel):
    user_id: str
    message: str
    embedding_model: str
    embedding_dimensions: int
    embedding: List[float]
    status: str


class ContextDocument(BaseModel):
    content: str
    # Database parameters (matching CargaConocimiento_iA schema)
    company_id: str
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    # Search metadata
    distance: float
    relevance_score: float


class RAGResponse(BaseModel):
    user_id: str
    message: str
    answer: str
    context_documents: List[ContextDocument]
    context_text: str
    total_documents_found: int
    embedding_dimensions: int
    collection_searched: str
    llm_model_used: str
    search_parameters: Dict[str, Any]
    status: str


class SearchDocument(BaseModel):
    content: str
    # Database parameters (matching CargaConocimiento_iA schema)
    company_id: str
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    # Search metadata
    distance: float
    relevance_score: float


class SearchResponse(BaseModel):
    user_id: str
    message: str
    documents: List[SearchDocument]
    total_found: int
    search_parameters: Dict[str, Any]
    embedding_dimensions: int
    status: str


def get_chat_service() -> ChatService:
    """Dependency injection for ChatService using the DI container."""
    return container.get_chat_service()

def get_rag_chat_service() -> ChatService:
    """Dependency injection for ChatService with Weaviate vectorstore for full RAG."""
    return container.get_chat_service_with_vectorstore()

def get_full_rag_dependencies():
    """Dependency injection for complete RAG with LLM answer generation."""
    return container.get_full_rag_chat_service()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: UnifiedRequest, dependencies: tuple = Depends(get_full_rag_dependencies)):
    """
    Clean chat endpoint with RAG-powered answer generation.
    
    Full RAG Flow:
    1. Generate embedding for user message
    2. Search relevant documents in Weaviate (using same flow as /search)
    3. Assemble context from retrieved documents
    4. Generate answer using LLM (Llama3) with context
    5. Return clean chat response with just the answer
    
    Returns only essential chat information, hiding RAG implementation details.
    """
    chat_service, llm_provider = dependencies
    
    # Process complete RAG query with LLM answer generation
    result = await chat_service.process_rag_query(
        user_id=request.user_id, 
        message=request.message,
        company_id=request.company_id,
        collection=request.collection,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        llm_provider=llm_provider
    )
    
    return ChatResponse(
        user_id=result["user_id"],
        message=result["message"], 
        answer=result["answer"],
        llm_model_used=result["llm_model_used"],
        status=result["status"]
    )


@router.post("/chat-test", response_model=EmbeddingTestResponse)
async def chat_test_endpoint(request: EmbeddingTestRequest, chat_service: ChatService = Depends(get_chat_service)):
    """
    Dedicated endpoint for testing embeddings model only.
    Returns detailed embedding information for testing purposes.
    Follows hexagonal architecture principles.
    """
    # Use ChatService to test embeddings with detailed response
    result = await chat_service.test_embedding(request.message, settings.embeddings_model_id)
    
    # Add user_id to result for consistent response
    result["user_id"] = request.user_id
    
    return EmbeddingTestResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: UnifiedRequest, chat_service: ChatService = Depends(get_rag_chat_service)):
    """
    Vector database search endpoint.
    
    Performs semantic search on the vector database using embeddings.
    Returns relevant documents with similarity scores.
    
    Flow:
    1. Convert query text to embedding
    2. Search vector database for similar documents
    3. Return ranked results with relevance scores
    
    Args:
        user_id: User identifier
        message: Search query text
        collection: Collection to search in (optional)
        top_k: Number of results to return (default: 5)
        similarity_threshold: Minimum similarity score (default: 0.7)
    """
    # Perform vector search using ChatService
    result = await chat_service.search_documents(
        query=request.message,  # Use 'message' field consistently
        company_id=request.company_id,
        collection=request.collection,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold
    )
    
    # Add user_id and message to result for consistent response
    result["user_id"] = request.user_id
    result["message"] = request.message
    
    return SearchResponse(**result)


@router.post("/chat-streaming")
async def chat_streaming_endpoint(request: UnifiedRequest, dependencies: tuple = Depends(get_full_rag_dependencies)):
    """
    Streaming chat endpoint with RAG-powered answer generation.
    
    Same functionality as /chat but with streaming response.
    Returns Server-Sent Events (SSE) format for real-time streaming.
    
    Response format:
    - metadata: Initial context information
    - chunk: Individual text chunks as they're generated
    - complete: Final completion signal
    """
    chat_service, llm_provider = dependencies
    
    async def generate_stream():
        async for chunk_data in chat_service.process_rag_query_stream(
            user_id=request.user_id,
            message=request.message,
            company_id=request.company_id,
            collection=request.collection,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            llm_provider=llm_provider
        ):
            # Send each chunk immediately as it arrives
            yield f"data: {json.dumps(chunk_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
