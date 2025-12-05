"""API endpoints for users management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.users_service import UsersService
from app.models.response_models import create_success_response, create_error_response
from app.models.user_models import CreateUserRequest, UpdateUserRequest, UpdateUserStatusRequest, UpdateUserPasswordRequest, UpdateUserAccessRequest
from app.utils.jwt_auth import get_current_user_with_company_validation, get_current_user

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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
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

    Creates a new user using SP_CREATE_USUARIO.
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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Create user using service (user_id is the current user creating the new user)
        results = await users_service.create_usuario(
            db=db,
            id_usuario=user_id,
            nuevo_usuario=request.nuevo_usuario,
            password=request.password,
            nombres=request.nombres,
            apellidos=request.apellidos,
            telefono=request.telefono or "",
            nuevo_rol=request.nuevo_rol,
            id_empresa=request.id_empresa,
            areas_string=request.areas_string
        )

        # SP may not return results for now, so we just treat empty results as success
        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in create_usuario: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=400,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
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


@router.put("/update_usuario")
async def update_usuario_endpoint(
    http_request: Request,
    request: UpdateUserRequest,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update user data endpoint.

    Updates user information using SP_UPDATE_DATOS_USUARIO.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateUserRequest with user data to update

    Returns:
        Dict with user update results including:
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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Update user using service (user_id is the admin performing the update)
        results = await users_service.update_usuario(
            db=db,
            id_admin=user_id,
            id_usuario=request.id_usuario,
            usuario=request.usuario,
            nombres=request.nombres,
            apellidos=request.apellidos,
            telefono=request.telefono
        )

        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in update_usuario: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=400,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )

        success_response = create_success_response("Usuario actualizado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_usuario endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.put("/update_usuario_status")
async def update_usuario_status_endpoint(
    http_request: Request,
    request: UpdateUserStatusRequest,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update user status endpoint.

    Updates user status (active/inactive) using SP_UPDATE_USUARIO_STATUS.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateUserStatusRequest with id_usuario and status

    Returns:
        Dict with user status update results including:
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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Update user status using service (user_id is the admin performing the update)
        results = await users_service.update_usuario_status(
            db=db,
            id_admin=user_id,
            id_usuario=request.id_usuario,
            status=request.status
        )

        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in update_usuario_status: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )

        success_response = create_success_response("Estado del usuario actualizado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_usuario_status endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.put("/update_usuario_password")
async def update_usuario_password_endpoint(
    http_request: Request,
    request: UpdateUserPasswordRequest,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update user password endpoint.

    Updates user password using SP_UPDATE_USUARIO_PASSWORD.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateUserPasswordRequest with id_usuario and clave_acceso

    Returns:
        Dict with user password update results including:
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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Update user password using service (user_id is the admin performing the update)
        results = await users_service.update_usuario_password(
            db=db,
            id_admin=user_id,
            id_usuario=request.id_usuario,
            clave_acceso=request.clave_acceso
        )

        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in update_usuario_password: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )

        success_response = create_success_response("Contraseña actualizada exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_usuario_password endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.put("/update_usuario_access")
async def update_usuario_access_endpoint(
    http_request: Request,
    request: UpdateUserAccessRequest,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Update user role and areas access endpoint.

    Updates user role and assigned areas using SP_UPDATE_USUARIO_ACCESS.
    Requires JWT authentication and validates company access.

    Args:
        request: UpdateUserAccessRequest with id_usuario, nuevo_rol, and areas_string

    Returns:
        Dict with user access update results including:
        - results: List of dictionaries with ID_TIPO_MENSAJE, MENSAJE
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 403 for access denied, 422 for validation errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        role_id = current_user.get('ID_TIPO_ROL')
        # Get company ID from request body (frontend sends it)
        id_empresa = request.id_empresa

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
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Validate company ID from request
        if not id_empresa or id_empresa <= 0:
            error_response = create_error_response("ID de empresa inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Update user access using service (user_id is the admin performing the update)
        results = await users_service.update_usuario_access(
            db=db,
            id_admin=user_id,
            id_usuario=request.id_usuario,
            nuevo_rol=request.nuevo_rol,
            id_empresa=id_empresa,
            areas_string=request.areas_string
        )

        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in update_usuario_access: {mensaje}")

            if tipo_mensaje == 1:
                raise HTTPException(
                    status_code=403,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )
            elif tipo_mensaje == 3:
                raise HTTPException(
                    status_code=422,
                    detail={"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}
                )

        success_response = create_success_response("Acceso del usuario actualizado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_usuario_access endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


# N8N Integration Endpoints
@router.post("/get_user_data_for_n8n")
async def get_user_data_for_n8n_endpoint(
    http_request: Request,
    users_service: UsersService = Depends(get_users_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get complete user data for n8n integration endpoint.

    Retrieves user information and all context needed for RAG queries using SP_GET_USER_DATA_FOR_N8N.
    Requires JWT authentication. Only accessible by users with role ID 4 (n8n/agent).

    Request body:
        {
            "telefono": str  // Phone number (max 15 chars)
        }

    Returns:
        List with:
        - On failure (1 result set):
            * {"ID_TIPO_MENSAJE": 1, "MENSAJE": "error message"}

        - On success (5 result sets):
            * {"ID_TIPO_MENSAJE": 2, "MENSAJE": "success message"}
            * {"ID_USUARIO": int, "USUARIO": str, ...}  // User data
            * {"ID_EMPRESA": int, "ID_AREA": int}       // Company and Area IDs
            * {"ID_IA_AREA": int}                        // IA Area config ID
            * {"ID_CHAT": int}                           // Existing chat ID (if any)

    Example success response:
        [
            {"ID_TIPO_MENSAJE": 2, "MENSAJE": "Usuario encontrado"},
            {"ID_USUARIO": 123, "USUARIO": "john_doe", "TELEFONO": "+51999888777"},
            {"ID_EMPRESA": 1, "ID_AREA": 2},
            {"ID_IA_AREA": 2},
            {"ID_CHAT": 4045}
        ]

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

        # Check if user has role ID 4 (agent role for N8N)
        if role_id != 4:
            raise HTTPException(
                status_code=403,
                detail={"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}
            )

        # Parse request body
        body = await http_request.json()
        telefono = body.get('telefono')

        # Validate input
        if not telefono or not isinstance(telefono, str) or len(telefono.strip()) == 0:
            error_response = create_error_response("Número de teléfono inválido")
            raise HTTPException(
                status_code=422,
                detail={"result": error_response.model_dump()}
            )

        # Get user by phone using service (user_id is the agent ID)
        results = await users_service.get_usuario_by_telefono(
            db=db,
            id_agente=user_id,
            telefono=telefono
        )

        if not results:
            error_response = create_error_response("Error al buscar usuario por teléfono")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        # Return only the results from the stored procedure (no extra fields)
        return results

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_usuario_by_telefono endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
