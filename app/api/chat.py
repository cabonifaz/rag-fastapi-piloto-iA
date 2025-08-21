from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from app.application.chat_service import ChatService
from app.core.config import settings
from app.core.container import container


router = APIRouter()


# Unified Request Schema for all endpoints
class UnifiedRequest(BaseModel):
    user_id: str
    message: str
    collection: str = None              # Optional, defaults to env config
    top_k: int = 5                     # Optional, for search endpoints
    similarity_threshold: float = 0.7   # Optional, for search endpoints


# Response Models
class ChatResponse(BaseModel):
    answer: str
    embedding: List[float]


class EmbeddingTestResponse(BaseModel):
    user_id: str
    message: str
    model_id: str
    embedding_dimensions: int
    embedding: List[float]
    status: str


class ContextDocument(BaseModel):
    content: str
    source: str
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
    llm_model_used: str = None
    search_parameters: Dict[str, Any]
    status: str


class SearchDocument(BaseModel):
    content: str
    source: str
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

@router.post("/chat", response_model=RAGResponse)
async def chat_endpoint(request: UnifiedRequest, dependencies: tuple = Depends(get_full_rag_dependencies)):
    """
    Complete RAG chat endpoint with answer generation.
    
    Full RAG Flow:
    1. Generate embedding for user message
    2. Search relevant documents in Weaviate (using same flow as /search)
    3. Assemble context from retrieved documents
    4. Generate answer using LLM (Llama3) with context
    5. Return complete response with answer and supporting documents
    
    Follows hexagonal architecture by using ChatService + LLM provider.
    """
    chat_service, llm_provider = dependencies
    
    # Process complete RAG query with LLM answer generation
    result = await chat_service.process_rag_query(
        user_id=request.user_id, 
        message=request.message,
        collection=request.collection,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        llm_provider=llm_provider
    )
    
    return RAGResponse(**result)


@router.post("/chat-test", response_model=EmbeddingTestResponse)
async def chat_test_endpoint(request: UnifiedRequest, chat_service: ChatService = Depends(get_chat_service)):
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
        collection=request.collection,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold
    )
    
    # Add user_id and message to result for consistent response
    result["user_id"] = request.user_id
    result["message"] = request.message
    
    return SearchResponse(**result)
