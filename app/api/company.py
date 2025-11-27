"""API endpoints for company management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.company_service import CompanyService
from app.models.response_models import create_success_response, create_error_response
from app.models.company_models import CompanyCreateRequest, CompanyStatusUpdateRequest
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_company_service() -> CompanyService:
    """Get singleton CompanyService from container."""
    return container.get_company_service()


@router.post("/create_company")
async def create_company_endpoint(
    request: CompanyCreateRequest,
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new company base endpoint.

    Creates a new company with associated roles and areas using SP_CREATE_EMPRESA_BASE.
    Requires JWT authentication. Uses user ID from JWT token.

    Args:
        request: CompanyCreateRequest with ruc and razon_social

    Returns:
        Dict with company creation results including:
        - results: List of dictionaries with id_rol, message, id_company, id_area
        - result: Success/error response

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
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

        # Create company using service
        results = await company_service.create_company(
            db=db,
            id_usuario=user_id,
            ruc=request.ruc,
            razon_social=request.razon_social
        )

        if not results:
            error_response = create_error_response("Error al crear la empresa")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if the stored procedure returned an error
        # ID_TIPO_MENSAJE = 1 indicates an error
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

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

        success_response = create_success_response("Empresa creada exitosamente")
        return {
            "results": results,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_company endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/get_companies")
async def get_companies_endpoint(
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all companies endpoint.

    Fetches all companies using SP_EMPRESAS_LST.
    Requires JWT authentication.

    Returns:
        Dict with:
        - companies: List of company dictionaries
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 500 for server errors
    """
    try:
        role_id = current_user.get('ID_TIPO_ROL')

        # Check if user is SuperAdmin (role_id = 1)
        if role_id != 1:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Get companies using service
        companies = await company_service.get_companies(db=db)

        success_response = create_success_response("Empresas obtenidas exitosamente")
        return {
            "companies": companies,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_companies endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/update_company_status")
async def update_company_status_endpoint(
    request: CompanyStatusUpdateRequest,
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update company status endpoint.

    Updates a company status using SP_UPDATE_EMPRESA_STATUS.
    Requires JWT authentication and SuperAdmin role.

    Args:
        request: CompanyStatusUpdateRequest with id_empresa and status

    Returns:
        Dict with:
        - results: List of dictionaries with update result information
        - result: Success/error response

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

        # Update company status using service
        results = await company_service.update_company_status(
            db=db,
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            status=request.status
        )

        if not results:
            error_response = create_error_response("Error al actualizar el estado de la empresa")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Check if the stored procedure returned an error
        # ID_TIPO_MENSAJE = 1 indicates an error
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

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

        success_response = create_success_response("Estado de empresa actualizado exitosamente")
        return {
            "results": results,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_company_status endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
