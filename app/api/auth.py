from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.container import container
from app.services.auth_service import AuthService
from app.models.user_models import LoginRequest, LoginResponse
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user
from pydantic import ValidationError
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_auth_service() -> AuthService:
    """Get singleton AuthService from container."""
    return container.get_auth_service()


@router.post("/login", response_model=LoginResponse)
async def login_endpoint(
    response: Response,
    login_request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
    db: Session = Depends(get_db)
):
    """
    User login endpoint

    Authenticates user credentials against SQL Server database.
    If ref (secret_key) is provided, validates company access.
    Returns user information and session details on successful login.

    Args:
        login_request: LoginRequest containing usuario, clave_acceso, and optional ref (secret_key)

    Returns:
        LoginResponse with user details and success status

    Raises:
        HTTPException: 401 for invalid credentials, 422 for validation errors, 500 for server errors
    """
    try:
        # Validate and authenticate user (with company if ref provided)
        login_response = await auth_service.authenticate_user(db, login_request)

        if not login_response:
            error_response = create_error_response("Credenciales o acceso a empresa inválidos")
            raise HTTPException(
                status_code=401,
                detail={"result": error_response.model_dump()}
            )

        # JWT token will be stored in frontend session storage
        # No cookie needed - token remains in response body
        return login_response

    except ValidationError as e:
        logger.error(f"Validation error in login endpoint: {e}")
        error_response = create_error_response(f"Datos de solicitud inválidos: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={"result": error_response.model_dump()}
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in login endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


class LogoutRequest(BaseModel):
    user_id: int

@router.post("/logout")
async def logout_endpoint(
    response: Response,
    logout_request: LogoutRequest,
    auth_service: AuthService = Depends(get_auth_service),
    db: Session = Depends(get_db)
):
    """
    User logout endpoint

    Updates user connection status to disconnected.

    Args:
        user_id: ID of the user to logout

    Returns:
        Success message

    Raises:
        HTTPException: 404 if user not found, 500 for server errors
    """
    try:
        success = await auth_service.logout_user(db, logout_request.user_id)
        
        if not success:
            error_response = create_error_response("Usuario no encontrado")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )
        
        # JWT token will be cleared from frontend session storage
        
        success_response = create_success_response("Sesión cerrada exitosamente")
        return {"result": success_response.model_dump()}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in logout endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/company-areas")
async def get_user_company_areas_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
    db: Session = Depends(get_db)
):
    """
    Get user company areas endpoint

    Retrieves company areas for the authenticated user using their user ID and role ID from JWT.

    Returns:
        List of company areas for the user

    Raises:
        HTTPException: 404 if no areas found, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        role_id = current_user.get('ID_TIPO_ROL')

        if not user_id or not role_id:
            error_response = create_error_response("Información de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        company_areas = await auth_service.get_user_company_areas(db, user_id, role_id)

        if company_areas is None:
            error_response = create_error_response("Error al obtener áreas de empresa")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Áreas de empresa obtenidas exitosamente")
        return {
            "company_areas": company_areas,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get company areas endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


