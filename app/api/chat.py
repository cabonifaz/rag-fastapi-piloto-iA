from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
import json
import logging
from app.utils.jwt_auth import get_current_user, get_current_user_with_company_validation
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.services.chat_service import ChatService
from app.core.config import settings
from app.core.container import container
from app.models.response_models import (
    MensajeResponse, 
    create_success_response, 
    create_error_response,
    create_warning_response
)

# Configure logging
logger = logging.getLogger(__name__)


router = APIRouter()


# Request Schema for endpoints that need company-specific search
class UnifiedRequest(BaseModel):
    user_id: str
    message: str
    company_id: str                         # Required, for company-specific search
    area: str                               # Required, for area-specific filtering and user role validation
    collection: str = None                   # Optional, defaults to env config
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None  # Optional, defaults to env config
    temperature: Optional[float] = None      # Optional, defaults to env config
    max_tokens: Optional[int] = None         # Optional, defaults to env config

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
    result: MensajeResponse


class EmbeddingTestResponse(BaseModel):
    user_id: str
    message: str
    embedding_model: str
    embedding_dimensions: int
    embedding: List[float]
    status: str
    result: MensajeResponse


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
    result: MensajeResponse


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
    result: MensajeResponse


def get_full_rag_dependencies():
    """Dependency injection for complete RAG with LLM answer generation."""
    return container.get_full_rag_chat_service()


@router.post("/chat-streaming")
async def chat_streaming_endpoint(
    request: UnifiedRequest,
    dependencies: tuple = Depends(get_full_rag_dependencies),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Streaming chat endpoint with RAG-powered answer generation.
    
    Same functionality as /chat but with streaming response.
    Returns Server-Sent Events (SSE) format for real-time streaming.
    
    Response format:
    - metadata: Initial context information
    - chunk: Individual text chunks as they're generated
    - complete: Final completion signal
    """
    try:
        chat_service, llm_provider = dependencies
        
        async def generate_stream():
            try:
                answer = ""
                async for chunk_data in chat_service.process_rag_query_stream(
                    user_id=request.user_id,
                    message=request.message,
                    company_id=request.company_id,
                    area=request.area,
                    collection=request.collection,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    if chunk_data["type"] == "chunk":
                        # Concatenate content
                        answer += chunk_data["content"]
                        # Send concatenated answer
                        yield f"data: {json.dumps({'type': 'chunk', 'content': answer})}\n\n"
                    else:
                        # Send metadata and complete as-is
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.error(f"AWS Client error in streaming: {error_code} - {e}")
                error_msg = "Error del servicio de modelo de lenguaje"
                if error_code == 'ValidationException':
                    error_msg = "Parámetros inválidos para el modelo de lenguaje"
                elif error_code == 'ThrottlingException':
                    error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
                error_response = create_error_response(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'result': error_response.model_dump()})}\n\n"
                
            except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
                logger.error(f"Connection error in streaming: {e}")
                error_response = create_error_response("Error de conexión del servicio")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Error de conexión del servicio', 'result': error_response.model_dump()})}\n\n"
                
            except TimeoutError as e:
                logger.error(f"Timeout error in streaming: {e}")
                error_response = create_error_response("Tiempo de espera de la solicitud agotado")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Tiempo de espera de la solicitud agotado', 'result': error_response.model_dump()})}\n\n"
                
            except Exception as e:
                logger.error(f"Unexpected error in streaming: {e}")
                error_response = create_error_response("Error interno del servidor")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Error interno del servidor', 'result': error_response.model_dump()})}\n\n"
        
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
        
    except ValidationError as e:
        logger.error(f"Validation error in streaming endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})
        
    except ValueError as e:
        logger.error(f"Value error in streaming endpoint: {e}")
        error_response = create_error_response(str(e))
        raise HTTPException(status_code=400, detail={"result": error_response.dict()})
        
    except Exception as e:
        logger.error(f"Unexpected error in streaming endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})
