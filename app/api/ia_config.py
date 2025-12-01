"""API endpoints for IA area configuration management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.ia_config_service import IaConfigService
from app.models.response_models import create_success_response, create_error_response
from app.models.ia_config_models import GetIaAreaConfigRequest, UpdateIaAreaConfigRequest, GetIaAreaConfigResponse
from app.utils.jwt_auth import get_current_user_with_company_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ia_config_service() -> IaConfigService:
    """Get singleton IaConfigService from container."""
    return container.get_ia_config_service()


@router.post("/get_ia_area_config")
async def get_ia_area_config_endpoint(
    request: GetIaAreaConfigRequest,
    ia_config_service: IaConfigService = Depends(get_ia_config_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Get IA area configuration endpoint.

    Retrieves full IA area configuration including models, parameters, and RAG settings
    using SP_GET_IA_AREA_CONFIG.
    Requires JWT authentication and validates company access.

    Args:
        request: GetIaAreaConfigRequest with id_empresa and id_area

    Returns:
        Dict with:
        - result: Configuration data with ID_IA_AREA, ID_AREA, ID_EMBEDDINGS, ID_LLM,
                  EMBEDDINGS_DIMENSIONS, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P,
                  RAG_TOP_K_RESULTS, RAG_SIMILARITY_THRESHOLD, RAG_ALPHA, ROLE_BEHAVIOR

    Raises:
        HTTPException: 404 if config not found, 403 for permission errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Get IA area config using service
        config = await ia_config_service.get_ia_area_config_full(
            db=db,
            id_area=request.id_area
        )

        if not config:
            error_response = create_error_response(f"No se encontró configuración para el área {request.id_area}")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )

        # Convert database column names to camelCase for API response
        return {
            "result": {
                "id_ia_area": config.get('ID_IA_AREA'),
                "id_area": config.get('ID_AREA'),
                "id_embeddings": config.get('ID_EMBEDDINGS'),
                "id_llm": config.get('ID_LLM'),
                "embeddings_dimensions": config.get('EMBEDDINGS_DIMENSIONS'),
                "llm_max_tokens": config.get('LLM_MAX_TOKENS'),
                "llm_temperature": config.get('LLM_TEMPERATURE'),
                "llm_top_p": config.get('LLM_TOP_P'),
                "rag_top_k_results": config.get('RAG_TOP_K_RESULTS'),
                "rag_similarity_threshold": config.get('RAG_SIMILARITY_THRESHOLD'),
                "rag_alpha": config.get('RAG_ALPHA'),
                "role_behavior": config.get('ROLE_BEHAVIOR')
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_ia_area_config endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/update_ia_area_config")
async def update_ia_area_config_endpoint(
    request: UpdateIaAreaConfigRequest,
    ia_config_service: IaConfigService = Depends(get_ia_config_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update IA area configuration endpoint.

    Updates IA area configuration including models, parameters, and RAG settings
    using SP_UPDATE_IA_AREA_BASE.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateIaAreaConfigRequest with all configuration parameters

    Returns:
        Dict with:
        - result: Success/error response with ID_TIPO_MENSAJE and message

    Raises:
        HTTPException: 400 for validation errors, 403 for permission/access errors, 500 for server errors
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

        # Update IA area config using service
        result = await ia_config_service.update_ia_area_config(
            db=db,
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            id_area=request.id_area,
            id_embeddings=request.id_embeddings,
            id_llm=request.id_llm,
            embeddings_dimensions=request.embeddings_dimensions,
            llm_max_tokens=request.llm_max_tokens,
            llm_temperature=request.llm_temperature,
            llm_top_p=request.llm_top_p,
            rag_top_k_results=request.rag_top_k_results,
            rag_similarity_threshold=request.rag_similarity_threshold,
            rag_alpha=request.rag_alpha,
            role_behavior=request.role_behavior
        )

        if not result:
            error_response = create_error_response("Error al actualizar la configuracion del area IA")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if the stored procedure returned an error
        # ID_TIPO_MENSAJE = 1 indicates an error
        if result and 'ID_TIPO_MENSAJE' in result:
            tipo_mensaje = result.get('ID_TIPO_MENSAJE')
            mensaje = result.get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje}: {mensaje}")

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

        return {
            "result": {
                "idTipoMensaje": result.get('ID_TIPO_MENSAJE', 2),
                "mensaje": result.get('MENSAJE', 'Configuracion del area IA actualizada exitosamente')
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_ia_area_config endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
