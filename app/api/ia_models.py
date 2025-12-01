"""API endpoints for IA models management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.ia_models_service import IAModelsService
from app.models.response_models import create_success_response, create_error_response
from app.models.ia_models_models import ModelCreateRequest
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ia_models_service() -> IAModelsService:
    """Get singleton IAModelsService from container."""
    return container.get_ia_models_service()


@router.get("/get_models")
async def get_models_endpoint(
    ia_models_service: IAModelsService = Depends(get_ia_models_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all available IA models endpoint.

    Fetches all available IA models using SP_MODELS_LST.
    Requires JWT authentication.

    Returns:
        Dict with:
        - models: List of model dictionaries
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Get models using service
        models = await ia_models_service.get_models(db=db)

        # SP_MODELS_LST does not return messages, only model data
        # Return models directly if retrieved, empty list if none found
        success_response = create_success_response("Modelos obtenidos exitosamente")
        return {
            "models": models if models else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_models endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/create_model")
async def create_model_endpoint(
    request: ModelCreateRequest,
    ia_models_service: IAModelsService = Depends(get_ia_models_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new IA model endpoint.

    Creates a new IA model using SP_CREATE_MODEL.
    Requires JWT authentication. Uses user ID from JWT token.

    Args:
        request: ModelCreateRequest with model details

    Returns:
        Dict with:
        - result: Success/error response with ID_TIPO_MENSAJE and message

    Raises:
        HTTPException: 400 for validation errors, 403 for permission errors, 500 for server errors
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

        # Check if user is SuperAdmin (role_id = 1)
        if role_id != 1:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Create model using service
        result = await ia_models_service.create_model(
            db=db,
            id_usuario=user_id,
            model=request.model,
            id_model=request.id_model,
            provider=request.provider,
            type_id=request.type_id,
            extra_parameter=request.extra_parameter
        )

        if not result:
            error_response = create_error_response("Error al crear el modelo")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if the stored procedure returned an error
        # ID_TIPO_MENSAJE = 1 indicates an error, 2 indicates success, 3 indicates validation error
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

        # Return success response
        return {
            "result": {
                "idTipoMensaje": result.get('ID_TIPO_MENSAJE', 2),
                "mensaje": result.get('MENSAJE', 'Modelo creado exitosamente')
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_model endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
