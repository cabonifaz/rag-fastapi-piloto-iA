"""API endpoints for area management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.area_service import AreaService
from app.models.response_models import create_success_response, create_error_response
from app.models.area_models import AreaCreateRequest
from app.utils.jwt_auth import get_current_user_with_company_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def get_area_service() -> AreaService:
    """Get singleton AreaService from container."""
    return container.get_area_service()


@router.post("/create_area")
async def create_area_endpoint(
    http_request: Request,
    request: AreaCreateRequest,
    area_service: AreaService = Depends(get_area_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Create a new area base endpoint.

    Creates a new area using SP_CREATE_AREA_BASE.
    Requires JWT authentication and validates company access.

    Args:
        request: AreaCreateRequest with id_empresa and area

    Returns:
        Dict with area creation results including:
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

        # Create area using service
        results = await area_service.create_area(
            db=db,
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            area=request.area
        )

        if not results:
            error_response = create_error_response("Error al crear el area")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if the stored procedure returned an error
        # ID_TIPO_MENSAJE = 1 indicates an error
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )

        success_response = create_success_response("Area creada exitosamente")
        return {
            "results": results,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_area endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/get_areas/{id_empresa}")
async def get_areas_endpoint(
    id_empresa: int,
    area_service: AreaService = Depends(get_area_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Get all areas for a company endpoint.

    Fetches all areas for a specific company using SP_AREAS_LST.
    Requires JWT authentication and validates company access.

    Args:
        id_empresa: Company ID from URL path

    Returns:
        Dict with:
        - areas: List of area dictionaries
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 403 for access denied, 500 for server errors
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
        if not isinstance(id_empresa, int) or id_empresa <= 0:
            error_response = create_error_response("ID de empresa inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Get areas using service
        areas = await area_service.get_areas(
            db=db,
            id_empresa=id_empresa
        )

        if areas is None:
            error_response = create_error_response("Error al obtener las areas")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Areas obtenidas exitosamente")
        return {
            "areas": areas,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_areas endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
