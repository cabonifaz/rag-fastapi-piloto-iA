"""API endpoints for knowledge/document management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging
import httpx

from app.core.database import get_db
from app.core.config import settings
from app.core.container import container
from app.services.knowledge_service import KnowledgeService
from app.models.response_models import create_success_response, create_error_response
from app.models.knowledge_models import GetKnowledgeRequest, UpdateKnowledgeStateRequest, BatchUploadKnowledgeRequest, BatchUpdateKnowledgeStateRequest, BatchDeleteKnowledgeRequest
from app.utils.jwt_auth import get_current_user_with_company_validation, get_current_user, JWTAuth

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
        blob_storage = container.get_blob_storage()
        service = KnowledgeService(db, blob_storage)
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
        blob_storage = container.get_blob_storage()
        service = KnowledgeService(db, blob_storage)
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

        # Extract message_result and created_ids from service response
        db_response = response['results']
        message_result = db_response.get('message_result') if isinstance(db_response, dict) else None
        created_ids = db_response.get('created_ids', []) if isinstance(db_response, dict) else []

        return {
            "uploads": response['uploads'],
            "message_result": message_result,
            "created_ids": created_ids,
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


@router.patch("/batch_update_knowledge_state")
async def batch_update_knowledge_state_endpoint(
    http_request: Request,
    request: BatchUpdateKnowledgeStateRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Batch update process state for multiple knowledge/documents endpoint.

    Updates the process state for multiple knowledge/document records in a single batch operation.
    Uses SP_CARGA_CONOC_ESTADO_PROCESO_BATCH_UPD for efficient batch update.
    Requires JWT authentication.

    Args:
        request: BatchUpdateKnowledgeStateRequest with id_cargas and id_estado_proceso

    Returns:
        Dict with:
        - message_result: Status message from the stored procedure
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 422 for validation errors, 500 for server errors
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

        # Validate id_cargas parameter
        if not isinstance(request.id_cargas, list) or len(request.id_cargas) == 0:
            error_response = create_error_response("Lista de IDs de carga inválida o vacía")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate each id_carga is a positive integer
        for id_carga in request.id_cargas:
            if not isinstance(id_carga, int) or id_carga <= 0:
                error_response = create_error_response("IDs de carga inválidos")
                raise HTTPException(
                    status_code=422,
                    detail={"result": error_response.model_dump()}
                )

        # Validate id_estado_proceso parameter
        if not isinstance(request.id_estado_proceso, int) or request.id_estado_proceso <= 0:
            error_response = create_error_response("ID de estado de proceso inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Batch update knowledge process state using service
        blob_storage = container.get_blob_storage()
        service = KnowledgeService(db, blob_storage)
        message_result = await service.batch_update_knowledge_state(
            id_usuario=user_id,
            id_cargas=request.id_cargas,
            id_estado_proceso=request.id_estado_proceso,
            usumod=current_user.get('USUARIO', 'System')
        )

        # Call n8n webhook if configured
        if settings.n8n_cc_webhook_url and settings.n8n_cc_jwt_secret:
            try:
                # Generate JWT token for n8n webhook
                n8n_token = JWTAuth.create_n8n_jwt_token(user_id, settings.n8n_cc_jwt_secret)

                # Call n8n webhook
                async with httpx.AsyncClient(timeout=10.0) as client:
                    webhook_response = await client.post(
                        settings.n8n_cc_webhook_url,
                        headers={
                            "Authorization": f"Bearer {n8n_token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "id_cargas": request.id_cargas,
                            "id_estado_proceso": request.id_estado_proceso,
                            "user_id": user_id
                        }
                    )
                    webhook_response.raise_for_status()
                    logger.info(f"n8n webhook called successfully for user {user_id}")
            except Exception as webhook_error:
                # Log error but don't fail the endpoint
                logger.error(f"Error calling n8n webhook: {webhook_error}")
                # Continue execution - webhook failure should not affect the main operation

        success_response = create_success_response("Estados de documentos actualizados exitosamente")
        return {
            "message_result": message_result,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in batch_update_knowledge_state endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.delete("")
async def batch_delete_knowledge_endpoint(
    http_request: Request,
    request: BatchDeleteKnowledgeRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Args:
        request: BatchDeleteKnowledgeRequest with id_cargas list

    Returns:
        Dict with:
        - message_result: Status message from the stored procedure
        - deleted_count: Number of records deleted from SQL Server
        - deleted_records: List of deleted records info
        - weaviate_result: Result of Weaviate deletion (success, deleted_count, errors)
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
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes para eliminar documentos"}}
            )

        # Validate id_cargas parameter
        if not isinstance(request.id_cargas, list) or len(request.id_cargas) == 0:
            error_response = create_error_response("Lista de IDs de carga inválida o vacía")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Validate each id_carga is a positive integer
        for id_carga in request.id_cargas:
            if not isinstance(id_carga, int) or id_carga <= 0:
                error_response = create_error_response("IDs de carga inválidos")
                raise HTTPException(
                    status_code=422,
                    detail={"result": error_response.model_dump()}
                )

        # Batch delete knowledge using service
        blob_storage = container.get_blob_storage()
        service = KnowledgeService(db, blob_storage)

        try:
            deletion_result = await service.batch_delete_knowledge(
                id_usuario=user_id,
                id_cargas=request.id_cargas,
                usumod=current_user.get('USUARIO', 'System')
            )
        except ValueError as ve:
            # Handle "not found" errors from Weaviate validation
            error_msg = str(ve)
            logger.warning(f"Documents not found in vector database: {error_msg}")
            error_response = create_error_response(error_msg)
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )

        if not deletion_result:
            error_response = create_error_response("Error al eliminar los documentos")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if rollback was performed (Weaviate failed, SQL was restored)
        if deletion_result.get('rollback_performed', False):
            # HTTP 409 Conflict - Partial failure with automatic rollback
            error_response = create_error_response(
                deletion_result.get('message_result', {}).get('MENSAJE',
                    'Eliminación de Weaviate falló. Los registros fueron restaurados automáticamente.')
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "message_result": deletion_result.get('message_result'),
                    "deleted_count": 0,  # No net deletion due to rollback
                    "restored_records": deletion_result.get('restored_records', []),
                    "weaviate_result": deletion_result.get('weaviate_result', []),
                    "rollback_performed": True,
                    "result": error_response.model_dump()
                }
            )

        success_response = create_success_response("Documentos eliminados exitosamente")

        return {
            "message_result": deletion_result.get('message_result'),
            "deleted_count": deletion_result.get('deleted_count', 0),
            "deleted_records": deletion_result.get('deleted_records', []),
            "weaviate_result": deletion_result.get('weaviate_result', []),
            "rollback_performed": False,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in batch_delete_knowledge endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
