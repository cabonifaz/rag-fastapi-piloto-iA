from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from typing import Dict, Any
import asyncio
import json
import logging
from sqlalchemy.orm import Session
from app.utils.jwt_auth import get_current_user, get_current_user_with_company_area_validation
from app.utils.agent_jwt_auth import get_current_agent_with_company_area_validation, get_current_agent_with_company_validation
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.services.rag_service import RagService
from app.domain.ports.llm_port import LLMPort
from app.core.config import settings
from app.core.container import container
from app.core.database import get_db
from app.models.response_models import (
    MensajeResponse,
    create_success_response,
    create_error_response,
    create_warning_response
)
from app.models.rag_models import UnifiedRequest, N8NRequest, N8NAnonymousRequest, N8NLLMOnlyRequest, N8NLLMOnlyAnonymousRequest, AgentStreamingRequest

# Configure logging
logger = logging.getLogger(__name__)


router = APIRouter()


def get_rag_service() -> RagService:
    """Get singleton RagService from container."""
    return container.get_rag_service()


def get_llm_provider() -> LLMPort:
    """Get singleton LLM provider from container."""
    return container.get_llm_provider()


@router.post("/chat-streaming")
async def chat_streaming_endpoint(
    request: UnifiedRequest,
    rag_service: RagService = Depends(get_rag_service),
    llm_provider: LLMPort = Depends(get_llm_provider),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation)
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

        async def generate_stream():
            try:
                answer = ""
                async for chunk_data in rag_service.process_rag_query_stream(
                    user_id=request.user_id,
                    message=request.message,
                    company_id=request.company_id,
                    area_id=request.area_id,
                    db=db,
                    created_at=request.created_at,
                    chat_id=request.chat_id,
                    request_timezone=request.request_timezone,
                    tts=request.tts
                ):
                    if chunk_data["type"] == "chunk":
                        # Concatenate text content
                        answer += chunk_data["content"]
                        output = f"data: {json.dumps({'type': 'text_chunk', 'content': answer})}\n\n"
                        yield output
                    elif chunk_data["type"] == "audio_chunk":
                        # Pass audio chunk as-is (base64 encoded)
                        output = f"data: {json.dumps({'type': 'audio_chunk', 'content': chunk_data['content']})}\n\n"
                        yield output
                    else:
                        # Send metadata, progress, complete, and errors as-is
                        output = f"data: {json.dumps(chunk_data)}\n\n"
                        yield output

                    # Force flush by yielding control back to event loop
                    await asyncio.sleep(0)
                        
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


@router.post("/chat-n8n")
async def chat_n8n_endpoint(
    request: N8NRequest,
    rag_service: RagService = Depends(get_rag_service),
    db: Session = Depends(get_db),
    current_agent: Dict[str, Any] = Depends(get_current_agent_with_company_area_validation)
):
    """
    Non-streaming chat endpoint for n8n integration with RAG-powered answer generation.

    Requires agent JWT authentication token in Authorization header.
    Validates agent access to requested company_id and area_id.

    Returns complete response in a single JSON object (no streaming).

    Response format:
    {
        "response": "Complete LLM response text",
        "result": {
            "idTipoMensaje": 2,  // 2 = success, 1 = error
            "mensaje": "Respuesta generada correctamente"
        }
    }

    Error format:
    {
        "result": {
            "idTipoMensaje": 1,
            "mensaje": "Error message"
        }
    }
    """
    try:
        # Validate role - only role 4 (Agente-IA) is allowed
        role_id = current_agent.get('ID_TIPO_ROL')
        if role_id != 4:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        agent_id = current_agent.get('ID_AGENTE')
        logger.info(f"[N8N REQUEST] Agent ID: {agent_id}, User ID: {request.user_id}, Message: {request.message}")

        # Call non-streaming RAG service
        result = await rag_service.process_rag_query_n8n(
            user_id=request.user_id,
            message=request.message,
            company_id=request.company_id,
            area_id=request.area_id,
            db=db,
            created_at=request.created_at,
            chat_id=request.chat_id,
            request_timezone=request.request_timezone
        )

        logger.info(f"[N8N RESPONSE] Result type: {result.get('result', {}).get('idTipoMensaje')}")

        # Check if the result indicates an error (idTipoMensaje = 1)
        if result.get("result", {}).get("idTipoMensaje") == 1:
            # Return error response with 400 status
            logger.error(f"[N8N ERROR] {result.get('result', {}).get('mensaje')}")
            raise HTTPException(status_code=400, detail=result)

        # Return successful response
        return result

    except HTTPException:
        # Re-raise HTTPException as-is
        raise

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Client error in n8n endpoint: {error_code} - {e}")
        error_msg = "Error del servicio de modelo de lenguaje"
        if error_code == 'ValidationException':
            error_msg = "Parámetros inválidos para el modelo de lenguaje"
        elif error_code == 'ThrottlingException':
            error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
        logger.error(f"Connection error in n8n endpoint: {e}")
        error_response = create_error_response("Error de conexión del servicio")
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except TimeoutError as e:
        logger.error(f"Timeout error in n8n endpoint: {e}")
        error_response = create_error_response("Tiempo de espera de la solicitud agotado")
        raise HTTPException(status_code=504, detail={"result": error_response.model_dump()})

    except ValueError as e:
        logger.error(f"Invalid input for n8n endpoint: {e}")
        error_msg = str(e)
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=400, detail={"result": error_response.model_dump()})

    except ValidationError as e:
        logger.error(f"Validation error in n8n endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

    except Exception as e:
        logger.error(f"Unexpected error in n8n endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})


@router.post("/chat-n8n-anonymous")
async def chat_n8n_anonymous_endpoint(
    request: N8NAnonymousRequest,
    rag_service: RagService = Depends(get_rag_service),
    db: Session = Depends(get_db),
    current_agent: Dict[str, Any] = Depends(get_current_agent_with_company_area_validation)
):
    """
    Non-streaming chat endpoint for n8n integration with RAG-powered answer generation for anonymous chats.

    Requires agent JWT authentication token in Authorization header.
    Validates agent access to requested company_id and area_id.

    Returns complete response in a single JSON object (no streaming).

    Response format:
    {
        "response": "Complete LLM response text",
        "result": {
            "idTipoMensaje": 2,  // 2 = success, 1 = error
            "mensaje": "Respuesta generada correctamente"
        }
    }

    Error format:
    {
        "result": {
            "idTipoMensaje": 1,
            "mensaje": "Error message"
        }
    }
    """
    try:
        # Validate role - only role 4 (Agente-IA) is allowed
        role_id = current_agent.get('ID_TIPO_ROL')
        if role_id != 4:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        agent_id = current_agent.get('ID_AGENTE')
        logger.info(f"[N8N ANONYMOUS REQUEST] Agent ID: {agent_id}, User Anonymous ID: {request.user_anonymous_id}, Message: {request.message}, RAG Query: {request.rag_query}")

        # Call non-streaming RAG service for anonymous chat
        result = await rag_service.process_rag_query_n8n_anonymous(
            user_anonymous_id=request.user_anonymous_id,
            message=request.message,
            company_id=request.company_id,
            area_id=request.area_id,
            db=db,
            created_at=request.created_at,
            chat_anonymous_id=request.chat_anonymous_id,
            rag_query=request.rag_query,
            request_timezone=request.request_timezone
        )

        logger.info(f"[N8N ANONYMOUS RESPONSE] Result type: {result.get('result', {}).get('idTipoMensaje')}")

        # Check if the result indicates an error (idTipoMensaje = 1)
        if result.get("result", {}).get("idTipoMensaje") == 1:
            # Return error response with 400 status
            logger.error(f"[N8N ANONYMOUS ERROR] {result.get('result', {}).get('mensaje')}")
            raise HTTPException(status_code=400, detail=result)

        # Return successful response
        return result

    except HTTPException:
        # Re-raise HTTPException as-is
        raise

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Client error in n8n anonymous endpoint: {error_code} - {e}")
        error_msg = "Error del servicio de modelo de lenguaje"
        if error_code == 'ValidationException':
            error_msg = "Parámetros inválidos para el modelo de lenguaje"
        elif error_code == 'ThrottlingException':
            error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
        logger.error(f"Connection error in n8n anonymous endpoint: {e}")
        error_response = create_error_response("Error de conexión del servicio")
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except TimeoutError as e:
        logger.error(f"Timeout error in n8n anonymous endpoint: {e}")
        error_response = create_error_response("Tiempo de espera de la solicitud agotado")
        raise HTTPException(status_code=504, detail={"result": error_response.model_dump()})

    except ValueError as e:
        logger.error(f"Invalid input for n8n anonymous endpoint: {e}")
        error_msg = str(e)
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=400, detail={"result": error_response.model_dump()})

    except ValidationError as e:
        logger.error(f"Validation error in n8n anonymous endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

    except Exception as e:
        logger.error(f"Unexpected error in n8n anonymous endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})


@router.post("/chat-n8n-llm-only")
async def chat_n8n_llm_only_endpoint(
    request: N8NLLMOnlyRequest,
    useGuidelines: bool = Query(True, description="Whether to use guidelines in the LLM response"),
    storeMessages: bool = Query(True, description="Whether to store messages in the database"),
    rag_service: RagService = Depends(get_rag_service),
    db: Session = Depends(get_db),
    current_agent: Dict[str, Any] = Depends(get_current_agent_with_company_validation)
):
    """
    Non-streaming chat endpoint for n8n integration with LLM-ONLY mode (no RAG).

    Requires agent JWT authentication token in Authorization header.
    Validates agent access to requested company_id (no area validation required).

    Query Parameters:
    - useGuidelines: bool (default: True) - Whether to use guidelines in the LLM response
    - storeMessages: bool (default: True) - Whether to store messages in the database

    Returns complete response in a single JSON object (no streaming).
    Uses LLM with conversation history only - no embeddings, no vector search, no RAG context.

    Response format:
    {
        "response": "Complete LLM response text",
        "result": {
            "idTipoMensaje": 2,  // 2 = success, 1 = error
            "mensaje": "Respuesta generada correctamente"
        }
    }

    Error format:
    {
        "result": {
            "idTipoMensaje": 1,
            "mensaje": "Error message"
        }
    }
    """
    try:
        # Validate role - only role 4 (Agente-IA) is allowed
        role_id = current_agent.get('ID_TIPO_ROL')
        if role_id != 4:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        agent_id = current_agent.get('ID_AGENTE')
        logger.info(f"[N8N LLM-ONLY REQUEST] Agent ID: {agent_id}, User ID: {request.user_id}, Message: {request.message}")

        # Call non-streaming LLM-only service (no RAG)
        result = await rag_service.process_llm_only_n8n(
            user_id=request.user_id,
            message=request.message,
            company_id=request.company_id,
            db=db,
            created_at=request.created_at,
            chat_id=request.chat_id,
            system_behavior=request.system_behavior,
            custom_llm=request.custom_llm,
            request_timezone=request.request_timezone,
            use_guidelines=useGuidelines,
            store_messages=storeMessages
        )

        logger.info(f"[N8N LLM-ONLY RESPONSE] Result type: {result.get('result', {}).get('idTipoMensaje')}")

        # Check if the result indicates an error (idTipoMensaje = 1)
        if result.get("result", {}).get("idTipoMensaje") == 1:
            # Return error response with 400 status
            logger.error(f"[N8N LLM-ONLY ERROR] {result.get('result', {}).get('mensaje')}")
            raise HTTPException(status_code=400, detail=result)

        # Return successful response
        return result

    except HTTPException:
        # Re-raise HTTPException as-is
        raise

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Client error in n8n LLM-only endpoint: {error_code} - {e}")
        error_msg = "Error del servicio de modelo de lenguaje"
        if error_code == 'ValidationException':
            error_msg = "Parámetros inválidos para el modelo de lenguaje"
        elif error_code == 'ThrottlingException':
            error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
        logger.error(f"Connection error in n8n LLM-only endpoint: {e}")
        error_response = create_error_response("Error de conexión del servicio")
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except TimeoutError as e:
        logger.error(f"Timeout error in n8n LLM-only endpoint: {e}")
        error_response = create_error_response("Tiempo de espera de la solicitud agotado")
        raise HTTPException(status_code=504, detail={"result": error_response.model_dump()})

    except ValueError as e:
        logger.error(f"Invalid input for n8n LLM-only endpoint: {e}")
        error_msg = str(e)
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=400, detail={"result": error_response.model_dump()})

    except ValidationError as e:
        logger.error(f"Validation error in n8n LLM-only endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

    except Exception as e:
        logger.error(f"Unexpected error in n8n LLM-only endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})


@router.post("/chat-n8n-llm-only-anonymous")
async def chat_n8n_llm_only_anonymous_endpoint(
    request: N8NLLMOnlyAnonymousRequest,
    useGuidelines: bool = Query(True, description="Whether to use guidelines in the LLM response"),
    storeMessages: bool = Query(True, description="Whether to store messages in the database"),
    rag_service: RagService = Depends(get_rag_service),
    db: Session = Depends(get_db),
    current_agent: Dict[str, Any] = Depends(get_current_agent_with_company_validation)
):
    """
    Non-streaming chat endpoint for n8n integration with LLM-ONLY mode for anonymous chats (no RAG).

    Requires agent JWT authentication token in Authorization header.
    Validates agent access to requested company_id (no area validation required).

    Query Parameters:
    - useGuidelines: bool (default: True) - Whether to use guidelines in the LLM response
    - storeMessages: bool (default: True) - Whether to store messages in the database

    Returns complete response in a single JSON object (no streaming).
    Uses LLM with conversation history only - no embeddings, no vector search, no RAG context.

    Response format:
    {
        "response": "Complete LLM response text",
        "result": {
            "idTipoMensaje": 2,  // 2 = success, 1 = error
            "mensaje": "Respuesta generada correctamente"
        }
    }

    Error format:
    {
        "result": {
            "idTipoMensaje": 1,
            "mensaje": "Error message"
        }
    }
    """
    try:
        # Validate role - only role 4 (Agente-IA) is allowed
        role_id = current_agent.get('ID_TIPO_ROL')
        if role_id != 4:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        agent_id = current_agent.get('ID_AGENTE')
        logger.info(f"[N8N LLM-ONLY ANONYMOUS REQUEST] Agent ID: {agent_id}, User Anonymous ID: {request.user_anonymous_id}, Message: {request.message}")

        # Call non-streaming LLM-only service for anonymous chat (no RAG)
        result = await rag_service.process_llm_only_n8n_anonymous(
            user_anonymous_id=request.user_anonymous_id,
            message=request.message,
            company_id=request.company_id,
            db=db,
            created_at=request.created_at,
            chat_anonymous_id=request.chat_anonymous_id,
            system_behavior=request.system_behavior,
            custom_llm=request.custom_llm,
            request_timezone=request.request_timezone,
            use_guidelines=useGuidelines,
            store_messages=storeMessages
        )

        logger.info(f"[N8N LLM-ONLY ANONYMOUS RESPONSE] Result type: {result.get('result', {}).get('idTipoMensaje')}")

        # Check if the result indicates an error (idTipoMensaje = 1)
        if result.get("result", {}).get("idTipoMensaje") == 1:
            # Return error response with 400 status
            logger.error(f"[N8N LLM-ONLY ANONYMOUS ERROR] {result.get('result', {}).get('mensaje')}")
            raise HTTPException(status_code=400, detail=result)

        # Return successful response
        return result

    except HTTPException:
        # Re-raise HTTPException as-is
        raise

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Client error in n8n LLM-only anonymous endpoint: {error_code} - {e}")
        error_msg = "Error del servicio de modelo de lenguaje"
        if error_code == 'ValidationException':
            error_msg = "Parámetros inválidos para el modelo de lenguaje"
        elif error_code == 'ThrottlingException':
            error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
        logger.error(f"Connection error in n8n LLM-only anonymous endpoint: {e}")
        error_response = create_error_response("Error de conexión del servicio")
        raise HTTPException(status_code=503, detail={"result": error_response.model_dump()})

    except TimeoutError as e:
        logger.error(f"Timeout error in n8n LLM-only anonymous endpoint: {e}")
        error_response = create_error_response("Tiempo de espera de la solicitud agotado")
        raise HTTPException(status_code=504, detail={"result": error_response.model_dump()})

    except ValueError as e:
        logger.error(f"Invalid input for n8n LLM-only anonymous endpoint: {e}")
        error_msg = str(e)
        error_response = create_error_response(error_msg)
        raise HTTPException(status_code=400, detail={"result": error_response.model_dump()})

    except ValidationError as e:
        logger.error(f"Validation error in n8n LLM-only anonymous endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

    except Exception as e:
        logger.error(f"Unexpected error in n8n LLM-only anonymous endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})


# @router.post("/agent-streaming")
# async def agent_streaming_endpoint(
#     request: AgentStreamingRequest,
#     rag_service: RagService = Depends(get_rag_service),
#     llm_provider: LLMPort = Depends(get_llm_provider),
#     db: Session = Depends(get_db),
#     current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation)
# ):
#     """
#     Agent-powered streaming rag endpoint with orchestrator analysis and RAG.
#     Uses agent orchestrator to analyze queries and determine workflow requirements
#     before processing with RAG-powered answer generation.
#     Requires external system authentication token for enhanced capabilities.
#     Returns Server-Sent Events (SSE) format for real-time streaming.

#     Response format:
#     - agent_analysis: Agent orchestrator's task breakdown and workflow analysis
#     - metadata: Initial context information including agent analysis
#     - chunk: Individual text chunks as they're generated
#     - complete: Final completion signal
#     """
#     try:

#         # Validate external token
#         if not request.external_token or not request.external_token.strip():
#             raise ValueError("External token is required for agent streaming")

#         async def generate_stream():
#             try:
#                 answer = ""
#                 async for chunk_data in rag_service.agent_orchestrator_stream(
#                     user_id=request.user_id,
#                     user=request.user,
#                     message=request.message,
#                     company_id=request.company_id,
#                     company=request.company,
#                     area_id=request.area_id,
#                     area=request.area,
#                     id_ia_area=request.id_ia_area,
#                     db=db,
#                     top_k=request.top_k,
#                     similarity_threshold=request.similarity_threshold,
#                     alpha=request.alpha,
#                     temperature=request.temperature,
#                     max_tokens=request.max_tokens,
#                     external_token=request.external_token
#                 ):
#                     if chunk_data["type"] == "chunk":
#                         # Concatenate content
#                         answer += chunk_data["content"]
#                         # Send concatenated answer
#                         yield f"data: {json.dumps({'type': 'chunk', 'content': answer})}\n\n"
#                     else:
#                         # Send metadata and complete as-is
#                         yield f"data: {json.dumps(chunk_data)}\n\n"

#                     # Force flush by yielding control back to event loop
#                     await asyncio.sleep(0)

#             except ClientError as e:
#                 error_code = e.response['Error']['Code']
#                 logger.error(f"AWS Client error in agent streaming: {error_code} - {e}")
#                 error_msg = "Error del servicio de modelo de lenguaje"
#                 if error_code == 'ValidationException':
#                     error_msg = "Parámetros inválidos para el modelo de lenguaje"
#                 elif error_code == 'ThrottlingException':
#                     error_msg = "Límite de tasa excedido. Por favor, inténtelo de nuevo más tarde"
#                 error_response = create_error_response(error_msg)
#                 yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'result': error_response.model_dump()})}\n\n"

#             except (NoCredentialsError, EndpointConnectionError, ConnectionError) as e:
#                 logger.error(f"Connection error in agent streaming: {e}")
#                 error_response = create_error_response("Error de conexión del servicio")
#                 yield f"data: {json.dumps({'type': 'error', 'message': 'Error de conexión del servicio', 'result': error_response.model_dump()})}\n\n"

#             except TimeoutError as e:
#                 logger.error(f"Timeout error in agent streaming: {e}")
#                 error_response = create_error_response("Tiempo de espera de la solicitud agotado")
#                 yield f"data: {json.dumps({'type': 'error', 'message': 'Tiempo de espera de la solicitud agotado', 'result': error_response.model_dump()})}\n\n"

#             except Exception as e:
#                 logger.error(f"Unexpected error in agent streaming: {e}")
#                 error_response = create_error_response("Error interno del servidor")
#                 yield f"data: {json.dumps({'type': 'error', 'message': 'Error interno del servidor', 'result': error_response.model_dump()})}\n\n"

#         return StreamingResponse(
#             generate_stream(),
#             media_type="text/event-stream",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "Access-Control-Allow-Origin": "*",
#                 "X-Accel-Buffering": "no",  # Disable nginx buffering
#             }
#         )

#     except ValidationError as e:
#         logger.error(f"Validation error in agent streaming endpoint: {e}")
#         error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
#         raise HTTPException(status_code=422, detail={"result": error_response.model_dump()})

#     except ValueError as e:
#         logger.error(f"Value error in agent streaming endpoint: {e}")
#         error_response = create_error_response(str(e))
#         raise HTTPException(status_code=400, detail={"result": error_response.dict()})

#     except Exception as e:
#         logger.error(f"Unexpected error in agent streaming endpoint: {e}")
#         error_response = create_error_response("Error interno del servidor")
#         raise HTTPException(status_code=500, detail={"result": error_response.model_dump()})