"""API endpoints for knowledge/document management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.services.knowledge_service import KnowledgeService
from app.models.response_models import create_success_response, create_error_response
from app.models.knowledge_models import GetKnowledgeRequest, UpdateKnowledgeStateRequest, BatchUploadKnowledgeRequest
from app.utils.jwt_auth import get_current_user_with_company_validation

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/get_knowledge")
async def get_knowledge_endpoint(
    http_request: Request,
    request: GetKnowledgeRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Get all knowledge/documents for a company endpoint.

    Fetches all knowledge/documents for a specific company and optionally a specific area
    using SP_CARGA_CONOCIMIENTO_EMPRESA_LST.
    Requires JWT authentication and validates company access.

    Args:
        request: GetKnowledgeRequest with id_empresa and optional id_area

    Returns:
        Dict with:
        - knowledge: List of knowledge/document dictionaries
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 403 for access denied, 422 for validation errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        role_id = current_user.get('ID_TIPO_ROL')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Check if user is SuperAdmin (role_id = 1) or Admin (role_id = 2)
        if role_id not in [1, 2]:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Validate id_empresa parameter
        if not isinstance(request.id_empresa, int) or request.id_empresa <= 0:
            error_response = create_error_response("ID de empresa inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate id_area parameter if provided
        if request.id_area is not None and (not isinstance(request.id_area, int) or request.id_area <= 0):
            error_response = create_error_response("ID de área inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Get knowledge using service
        service = KnowledgeService(db)
        knowledge_list = await service.get_knowledge_by_company(
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            id_area=request.id_area
        )

        if knowledge_list is None:
            error_response = create_error_response("Error al obtener los documentos")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Documentos obtenidos exitosamente")
        return {
            "knowledge": knowledge_list,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_knowledge endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.patch("/update_knowledge_state")
async def update_knowledge_state_endpoint(
    http_request: Request,
    request: UpdateKnowledgeStateRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update knowledge/document process state endpoint.

    Updates the process state for a knowledge/document record using SP_CARGA_CONOC_ESTADO_PROCESO_UPD.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateKnowledgeStateRequest with id_carga and id_estado_proceso

    Returns:
        Dict with knowledge update state results including:
        - results: List of dictionaries with ID_TIPO_MENSAJE, MENSAJE
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 403 for access denied, 422 for validation errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        role_id = current_user.get('ID_TIPO_ROL')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Check if user is SuperAdmin (role_id = 1) or Admin (role_id = 2)
        if role_id not in [1, 2]:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Validate id_carga parameter
        if not isinstance(request.id_carga, int) or request.id_carga <= 0:
            error_response = create_error_response("ID de carga inv�lido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate id_estado_proceso parameter
        if not isinstance(request.id_estado_proceso, int) or request.id_estado_proceso <= 0:
            error_response = create_error_response("ID de estado de proceso inv�lido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Update knowledge process state using service (user_id is the one updating)
        service = KnowledgeService(db)
        results = await service.repository.update_knowledge_process_state(
            id_carga=request.id_carga,
            id_estado_proceso=request.id_estado_proceso,
            usumod=current_user.get('USUARIO', 'System')
        )

        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in update_knowledge_state: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )

        success_response = create_success_response("Estado del documento actualizado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_knowledge_state endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/batch_upload_knowledge")
async def batch_upload_knowledge_endpoint(
    http_request: Request,
    request: BatchUploadKnowledgeRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Batch upload multiple knowledge/documents endpoint.

    Creates knowledge records for multiple PDFs and generates presigned S3 URLs for upload.
    Uses SP_CARGA_CONOCIMIENTO_BATCH_INS for efficient batch insert.
    Requires JWT authentication and validates company access.

    Args:
        request: BatchUploadKnowledgeRequest with id_empresa, id_area, pdf_keys, and optional id_modelo_embedding

    Returns:
        Dict with:
        - uploads: List of upload objects with presigned URLs
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 403 for access denied, 422 for validation errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        role_id = current_user.get('ID_TIPO_ROL')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Check if user is SuperAdmin (role_id = 1) or Admin (role_id = 2)
        if role_id not in [1, 2]:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Validate id_empresa parameter
        if not isinstance(request.id_empresa, int) or request.id_empresa <= 0:
            error_response = create_error_response("ID de empresa inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate id_area parameter
        if not isinstance(request.id_area, int) or request.id_area <= 0:
            error_response = create_error_response("ID de área inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate pdf_keys parameter
        if not isinstance(request.pdf_keys, list) or len(request.pdf_keys) == 0:
            error_response = create_error_response("Lista de archivos PDF inválida o vacía")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate each pdf key is a non-empty string
        for pdf_key in request.pdf_keys:
            if not isinstance(pdf_key, str) or len(pdf_key) == 0:
                error_response = create_error_response("Nombres de archivo PDF inválidos")
                raise HTTPException(
                    status_code=422,
                    detail={"result": error_response.model_dump()}
                )

        # Batch upload knowledge using service
        service = KnowledgeService(db)
        response = await service.generate_presigned_urls(
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            id_area=request.id_area,
            pdf_keys=request.pdf_keys,
            id_modelo_embedding=request.id_modelo_embedding
        )

        if not response or not response.get('uploads'):
            error_response = create_error_response("Error al generar URLs para los documentos")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("URLs de carga generadas exitosamente")
        return {
            "uploads": response['uploads'],
            "results": response['results'],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in batch_upload_knowledge endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
