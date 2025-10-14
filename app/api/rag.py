from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from typing import Dict, Any
import json
import logging
from sqlalchemy.orm import Session
from app.utils.jwt_auth import get_current_user, get_current_user_with_company_validation
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.services.rag_service import RagService
from app.core.config import settings
from app.core.container import container
from app.core.database import get_db
from app.models.response_models import (
    MensajeResponse,
    create_success_response,
    create_error_response,
    create_warning_response
)
from app.models.rag_models import UnifiedRequest, AgentStreamingRequest

# Configure logging
logger = logging.getLogger(__name__)


router = APIRouter()


def get_full_rag_dependencies(db: Session = Depends(get_db)):
    """Dependency injection for complete RAG with LLM answer generation."""
    rag_service, llm_provider = container.get_full_rag_chat_service()
    rag_service.db = db
    return rag_service, llm_provider


@router.post("/chat-streaming")
async def chat_streaming_endpoint(
    request: UnifiedRequest,
    dependencies: tuple = Depends(get_full_rag_dependencies),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Streaming chat endpoint with RAG-powered answer generation.
    
    Returns Server-Sent Events (SSE) format for real-time streaming.
    
    Response format:
    - metadata: Initial context information
    - chunk: Individual text chunks as they're generated
    - complete: Final completion signal
    """
    try:
        print("=== FULL REQUEST AS DICT ===")
        print(request.model_dump())
        print("============================")
        rag_service, llm_provider = dependencies
        
        async def generate_stream():
            try:
                answer = ""
                async for chunk_data in rag_service.process_rag_query_stream(
                    user_id=request.user_id,
                    user=request.user,
                    message=request.message,
                    company_id=request.company_id,
                    company=request.company,
                    area_id=request.area_id,
                    area=request.area,
                    id_ia_area=request.id_ia_area,
                    created_at=created_at,
                    chat_id=chat_id,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    alpha=request.alpha,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    if chunk_data["type"] == "chunk":
                        # Concatenate content
                        answer += chunk_data["content"]
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

            except ValueError as e:
                logger.error(f"Invalid input for LLM: {e}")
                error_msg = str(e)
                error_response = create_error_response(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'result': error_response.model_dump()})}\n\n"

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


@router.post("/agent-streaming")
async def agent_streaming_endpoint(
    request: AgentStreamingRequest,
    dependencies: tuple = Depends(get_full_rag_dependencies),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Agent-powered streaming rag endpoint with orchestrator analysis and RAG.
    Uses agent orchestrator to analyze queries and determine workflow requirements
    before processing with RAG-powered answer generation.
    Requires external system authentication token for enhanced capabilities.
    Returns Server-Sent Events (SSE) format for real-time streaming.

    Response format:
    - agent_analysis: Agent orchestrator's task breakdown and workflow analysis
    - metadata: Initial context information including agent analysis
    - chunk: Individual text chunks as they're generated
    - complete: Final completion signal
    """
    try:
        rag_service, llm_provider = dependencies

        # Validate external token
        if not request.external_token or not request.external_token.strip():
            raise ValueError("External token is required for agent streaming")

        async def generate_stream():
            try:
                answer = ""
                async for chunk_data in rag_service.agent_orchestrator_stream(
                    user_id=request.user_id,
                    user=request.user,
                    message=request.message,
                    company_id=request.company_id,
                    company=request.company,
                    area_id=request.area_id,
                    area=request.area,
                    id_ia_area=request.id_ia_area,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    alpha=request.alpha,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    external_token=request.external_token
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
                logger.error(f"AWS Client error in agent streaming: {error_code} - {e}")
                error_msg = "Error del servicio de modelo de lenguaje"
                if error_code == 'ValidationException':
                    error_msg = "Parámetros inválidos para el modelo de lenguaje"
                elif error_code == 'ThrottlingException':
                    error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
                error_response = create_error_response(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'result': error_response.model_dump()})}\n\n"

            except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
                logger.error(f"Connection error in agent streaming: {e}")
                error_response = create_error_response("Error de conexión del servicio")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Error de conexión del servicio', 'result': error_response.model_dump()})}\n\n"

            except TimeoutError as e:
                logger.error(f"Timeout error in agent streaming: {e}")
                error_response = create_error_response("Tiempo de espera de la solicitud agotado")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Tiempo de espera de la solicitud agotado', 'result': error_response.model_dump()})}\n\n"

            except Exception as e:
                logger.error(f"Unexpected error in agent streaming: {e}")
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
        logger.error(f"Validation error in agent streaming endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

    except ValueError as e:
        logger.error(f"Value error in agent streaming endpoint: {e}")
        error_response = create_error_response(str(e))
        raise HTTPException(status_code=400, detail={"result": error_response.dict()})

    except Exception as e:
        logger.error(f"Unexpected error in agent streaming endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})