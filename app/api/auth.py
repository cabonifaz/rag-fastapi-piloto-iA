from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.auth_service import AuthService
from app.models.user_models import LoginRequest, LoginResponse, UserInfo
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
            print(f"*** LOGIN FAILED - AUTH SERVICE RETURNED NONE FOR: {login_request.usuario} ***")
            error_response = create_error_response("Credenciales inválidas")
            raise HTTPException(
                status_code=401,
                detail={"result": error_response.model_dump()}
            )
        
        # Set HttpOnly cookie with JWT token (8 hours expiration)
        print(f"*** SETTING JWT COOKIE FOR USER: {login_request.usuario} ***")
        print(f"*** COOKIE DOMAIN: Default (current domain) ***")
        print(f"*** COOKIE PATH: / ***")
        print(f"*** COOKIE SECURE: False ***")
        print(f"*** COOKIE SAMESITE: lax ***")
        
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
        
        print(f"*** USER LOGIN SUCCESSFUL: {login_request.usuario} ***")
        print(f"*** LOGIN RESPONSE BEING SENT: {login_response.model_dump()} ***")
        logger.info(f"User login successful: {login_request.usuario}")
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
        print(f"*** UNEXPECTED ERROR IN LOGIN ENDPOINT: {e} ***")
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
        print(f"*** CLEARING JWT COOKIE FOR USER ID: {logout_request.user_id} ***")
        response.set_cookie(
            key="jwt_token",
            value="",
            max_age=0,  # Expire immediately
            httponly=True,
            secure=False,   # Set to True in production with HTTPS
            samesite="lax"  # Less strict for development
        )
        print("*** JWT COOKIE CLEARED ***")
        
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
        print(f"*** JWT VALIDATION ENDPOINT - USER ID: {current_user.get('ID_USUARIO')} ***")
        
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