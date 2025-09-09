from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.auth_service import AuthService
from app.models.user_models import LoginRequest, LoginResponse, UserInfo, RefreshCompanyAreaRequest
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user
from pydantic import ValidationError
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """Dependency injection for AuthService"""
    return AuthService(db)


@router.post("/login", response_model=LoginResponse)
async def login_endpoint(
    response: Response,
    login_request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    User login endpoint
    
    Authenticates user credentials against SQL Server database.
    Returns user information and session details on successful login.
    
    Args:
        login_request: LoginRequest containing usuario and clave_acceso
        
    Returns:
        LoginResponse with user details and success status
        
    Raises:
        HTTPException: 401 for invalid credentials, 422 for validation errors, 500 for server errors
    """
    try:
        # Validate and authenticate user
        login_response = await auth_service.authenticate_user(login_request)
        
        if not login_response:
            error_response = create_error_response("Credenciales inválidas")
            raise HTTPException(
                status_code=401,
                detail={"result": error_response.model_dump()}
            )
        
        # Set HttpOnly cookie with JWT token (8 hours expiration)
        
        response.set_cookie(
            key="jwt_token",
            value=login_response.token,
            max_age=8 * 60 * 60,  # 8 hours in seconds
            httponly=True,  # Can't be accessed via JavaScript (XSS protection)
            secure=False,   # Set to True in production with HTTPS
            samesite="lax",  # Less strict for development
            path="/"  # Explicitly set path
        )
        
        # Remove token from response body for security
        login_response.token = None
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
    auth_service: AuthService = Depends(get_auth_service)
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
        success = await auth_service.logout_user(logout_request.user_id)
        
        if not success:
            error_response = create_error_response("Usuario no encontrado")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )
        
        # Clear the HttpOnly cookie
        response.set_cookie(
            key="jwt_token",
            value="",
            max_age=0,  # Expire immediately
            httponly=True,
            secure=False,   # Set to True in production with HTTPS
            samesite="lax"  # Less strict for development
        )
        
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


@router.get("/user/{user_id}", response_model=UserInfo)
async def get_user_info_endpoint(
    user_id: int,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get user information endpoint
    
    Retrieves user details by user ID.
    
    Args:
        user_id: ID of the user to retrieve
        
    Returns:
        UserInfo with user details
        
    Raises:
        HTTPException: 404 if user not found, 500 for server errors
    """
    try:
        user_info = await auth_service.get_user_info(user_id)
        
        if not user_info:
            error_response = create_error_response("Usuario no encontrado")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )
        
        return user_info
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in get user info endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/validate")
async def validate_jwt_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    JWT validation endpoint for route guarding.
    
    Simply validates that the JWT cookie is present and valid.
    Returns user info if valid, 401 if not.
    
    This is used by GuardRoute to protect frontend routes.
    """
    try:
        
        success_response = create_success_response("JWT válido")
        return {
            "valid": True,
            "user_id": current_user.get('ID_USUARIO'),
            "result": success_response.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in validate endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/refresh-company-area")
async def refresh_company_area_endpoint(
    request: Request,
    response: Response,
    refresh_request: RefreshCompanyAreaRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh JWT token with new company/area information
    
    Updates the JWT cookie with new company/area values while maintaining
    the same expiration time and other user information.
    
    Args:
        refresh_request: New company/area information
        current_user: Current authenticated user from JWT
        
    Returns:
        Success response if refresh successful
        
    Raises:
        HTTPException: 400 for invalid company/area, 401 for auth errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        if not user_id:
            error_response = create_error_response("Usuario no válido")
            raise HTTPException(
                status_code=401,
                detail={"result": error_response.model_dump()}
            )
        
        # Validate that JWT contains valid company/area parameters
        current_empresa = current_user.get('ID_EMPRESA')
        current_area = current_user.get('ID_AREA')
        
        if current_empresa is None or current_area is None:
            error_response = create_error_response("JWT no contiene información válida de empresa/área")
            raise HTTPException(
                status_code=401,
                detail={"result": error_response.model_dump()}
            )
        
        logger.info(f"User {user_id} requesting company/area change from {current_empresa}/{current_area} to {refresh_request.id_empresa}/{refresh_request.id_area}")
        
        # Create new JWT token with updated company/area
        new_jwt_token = await auth_service.refresh_company_area_jwt(user_id, refresh_request)
        
        if not new_jwt_token:
            error_response = create_error_response("No tienes acceso a la empresa/área seleccionada")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )
        
        # Update HttpOnly cookie with new JWT token (same settings as login)
        response.set_cookie(
            key="jwt_token",
            value=new_jwt_token,
            max_age=8 * 60 * 60,  # 8 hours in seconds
            httponly=True,  # Can't be accessed via JavaScript (XSS protection)
            secure=False,   # Set to True in production with HTTPS
            samesite="lax",  # Less strict for development
            path="/"  # Explicitly set path
        )
        
        success_response = create_success_response("Empresa/área actualizada exitosamente")
        return {"result": success_response.model_dump()}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in refresh company/area endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )