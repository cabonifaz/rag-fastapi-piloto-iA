from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from app.application.chat_service import ChatService
from app.core.config import settings
from app.core.container import container


router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    embedding: List[float]


class EmbeddingTestResponse(BaseModel):
    message: str
    model_id: str
    embedding_dimensions: int
    embedding: List[float]
    status: str


class ContextDocument(BaseModel):
    content: str
    source: str
    distance: float


class RAGResponse(BaseModel):
    user_id: str
    message: str
    context_documents: List[ContextDocument]
    context_text: str
    total_documents_found: int
    embedding_dimensions: int
    status: str


def get_chat_service() -> ChatService:
    """Dependency injection for ChatService using the DI container."""
    return container.get_chat_service()

def get_rag_chat_service() -> ChatService:
    """Dependency injection for ChatService with Weaviate vectorstore for full RAG."""
    return container.get_chat_service_with_vectorstore()

@router.post("/chat", response_model=RAGResponse)
async def chat_endpoint(request: ChatRequest, chat_service: ChatService = Depends(get_rag_chat_service)):
    """
    Full RAG chat endpoint with Weaviate vector search.
    
    Flow:
    1. Generate embedding for user message
    2. Search relevant documents in Weaviate
    3. Return context documents and assembled context
    
    Follows hexagonal architecture by using ChatService.
    """
    # Process RAG query with full pipeline
    result = await chat_service.process_rag_query(request.user_id, request.message)
    
    return RAGResponse(**result)


@router.post("/chat-test", response_model=EmbeddingTestResponse)
async def chat_test_endpoint(request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """
    Dedicated endpoint for testing embeddings model only.
    Returns detailed embedding information for testing purposes.
    Follows hexagonal architecture principles.
    """
    # Use ChatService to test embeddings with detailed response
    result = await chat_service.test_embedding(request.message, settings.embeddings_model_id)
    
    return EmbeddingTestResponse(**result)
