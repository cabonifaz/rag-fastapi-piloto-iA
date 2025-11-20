"""API endpoints for users management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.users_service import UsersService
from app.models.response_models import create_success_response, create_error_response
from app.models.user_models import CreateUserRequest
from app.utils.jwt_auth import get_current_user_with_company_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def get_users_service() -> UsersService:
    """Get singleton UsersService from container."""
    return container.get_users_service()


@router.get("/get_usuarios/{id_empresa}")
async def get_usuarios_endpoint(
    id_empresa: int,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Get all users for a company endpoint.

    Fetches all users for a specific company using SP_USUARIOS_LST.
    Requires JWT authentication and validates company access.

    Args:
        id_empresa: Company ID from URL path

    Returns:
        Dict with:
        - usuarios: List of user dictionaries
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

        # Get usuarios using service
        usuarios = await users_service.get_usuarios(
            db=db,
            id_empresa=id_empresa
        )

        if usuarios is None:
            error_response = create_error_response("Error al obtener los usuarios")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Usuarios obtenidos exitosamente")
        return {
            "usuarios": usuarios,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_usuarios endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/create_usuario")
async def create_usuario_endpoint(
    http_request: Request,
    request: CreateUserRequest,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Create a new user endpoint.

    Creates a new user using SP_USUARIO_CREATE.
    Requires JWT authentication and validates company access.

    Args:
        request: CreateUserRequest with user details and areas

    Returns:
        Dict with user creation results including:
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

        # Create user using service
        results = await users_service.create_usuario(
            db=db,
            nuevo_usuario=request.nuevo_usuario,
            password=request.password,
            nombres=request.nombres,
            apellidos=request.apellidos,
            id_tipo_rol=request.id_tipo_rol,
            id_empresa=request.id_empresa,
            areas_string=request.areas_string
        )

        # SP may not return results for now, so we just treat empty results as success
        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )

        success_response = create_success_response("Usuario creado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_usuario endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
