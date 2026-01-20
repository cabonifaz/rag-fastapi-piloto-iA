"""API endpoints for company management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.company_service import CompanyService
from app.models.response_models import create_success_response, create_error_response
from app.models.company_models import CompanyCreateRequest, CompanyStatusUpdateRequest, CompanyLogoUploadRequest
from app.utils.jwt_auth import get_current_user
from app.utils.cache import SimpleCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Global cache instance for companies
companies_cache = SimpleCache()

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

        # Invalidate companies cache after successful creation
        companies_cache.clear("companies_login")

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

        # Invalidate companies cache after successful status update
        companies_cache.clear("companies_login")

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


@router.get("/get_companies_login")
async def get_companies_login_endpoint(
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db)
):
    """
    Get companies with login credentials endpoint.

    Fetches all companies with their secret keys using SP_EMPRESAS_LST_LOGIN.
    Uses in-memory cache with 1 day TTL, invalidated when companies are created/updated.

    Returns:
        List of dictionaries with RAZON_SOCIAL and SECRET_KEY
    """
    # Define cache key and TTL
    cache_key = "companies_login"
    cache_ttl = 86400  # 1 day (invalidated on company create/update)

    # Try to get from cache first
    cached_result = companies_cache.get(cache_key, ttl_seconds=cache_ttl)
    if cached_result is not None:
        return cached_result

    # Cache miss - fetch from database
    result = await company_service.get_companies_login(db=db)

    # Store in cache
    companies_cache.set(cache_key, result, ttl_seconds=cache_ttl)

    return result


@router.get("/get_companies_paginated")
async def get_companies_paginated_endpoint(
    page: int = Query(default=1, ge=1, description="Número de página (mínimo 1)"),
    page_size: int = Query(default=10, ge=1, le=100, description="Filas por página (1-100)"),
    search: Optional[str] = Query(default=None, max_length=200, description="Término de búsqueda"),
    order_field: str = Query(
        default='RAZON_SOCIAL',
        regex='^(ID_EMPRESA|RUC|RAZON_SOCIAL|FCHCRE|ID_ESTADO_REGISTRO)$',
        description="Campo de ordenamiento"
    ),
    order_direction: str = Query(
        default='ASC',
        regex='^(ASC|DESC)$',
        description="Dirección de ordenamiento"
    ),
    status_filter: Optional[int] = Query(default=None, ge=0, le=1, description="Filtro de estado: 0=inactivo, 1=activo, null=todos"),  
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get paginated companies endpoint.

    Fetches paginated companies using SP_EMPRESAS_LST_PAG with server-side pagination,
    sorting, and search. Requires JWT authentication and validates role permissions.

    Query Parameters:
        page: Page number (starting at 1)
        page_size: Number of items per page (1-100)
        search: Optional search term for RAZON_SOCIAL or RUC
        order_field: Field to sort by (default 'RAZON_SOCIAL')
        order_direction: Sort direction ASC or DESC (default 'ASC')
        status_filter: Filter by status (0=inactive, 1=active, null=all)

    Returns:
        Dict with:
        - data: List of company dictionaries
        - pagination: Object with total_records, current_page, page_size, total_pages
        - result: Success/error response

    Raises:
        HTTPException: 400 for validation errors, 403 for permission errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        
        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Call service with Spanish parameter names
        result = await company_service.get_companies_paginated(
            db=db,
            id_usuario=user_id,
            num_pagina=page,
            tam_pagina=page_size,
            term_busqueda=search if search else None,
            campo_orden=order_field,
            dir_orden=order_direction,
            filtro_estado=status_filter
        )

        # ✅ Check if there's a message from the SP
        if result.get('message_result'):  # ✅ Cambio aquí: message_result en vez de results
            message_result = result['message_result']
            tipo_mensaje = message_result.get('ID_TIPO_MENSAJE')
            mensaje = message_result.get('MENSAJE', 'Error desconocido')
            
            
            if tipo_mensaje == 2:
                logger.info(f"SP returned success message: {mensaje}")
        
            
            elif tipo_mensaje == 1:
                logger.warning(f"SP returned business error: {mensaje}")
                raise HTTPException(
                    status_code=403,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )
                     
            elif tipo_mensaje == 3:
                logger.error(f"SP returned technical error: {mensaje}")
                raise HTTPException(
                    status_code=422,
                    detail={"result": {"idTipoMensaje": tipo_mensaje, "mensaje": mensaje}}
                )

        success_response = create_success_response("Empresas obtenidas exitosamente")
        return {
            "data": result.get('data', []),
            "pagination": result.get('pagination', {}),
            "result": success_response.model_dump()
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_companies_paginated endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )

@router.post("/upload_logo")
async def upload_company_logo_endpoint(
    request: CompanyLogoUploadRequest,
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate presigned URL for company logo upload endpoint.

    Generates presigned S3 URL and updates database with logo path.
    Frontend then uploads the logo file directly to S3 using the presigned URL.
    Requires JWT authentication and SuperAdmin role.

    Args:
        request: CompanyLogoUploadRequest with id_empresa and logo_filename

    Returns:
        Dict with:
        - presigned_url: S3 presigned PUT URL (5 min expiration)
        - s3_key: S3 object key path
        - logo_filename: Original filename
        - results: DB update results
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

        # Generate presigned URL and update database
        result = await company_service.generate_logo_presigned_url(
            db=db,
            id_usuario=user_id,
            id_empresa=request.id_empresa,
            logo_filename=request.logo_filename
        )

        # Check if the stored procedure returned an error
        if result.get('results') and 'ID_TIPO_MENSAJE' in result['results'][0]:
            tipo_mensaje = result['results'][0].get('ID_TIPO_MENSAJE')
            mensaje = result['results'][0].get('MENSAJE', 'Error desconocido')

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

        # Invalidate companies cache after successful logo upload
        companies_cache.clear("companies_login")

        success_response = create_success_response("Presigned URL generada exitosamente")
        return {
            "presigned_url": result['presigned_url'],
            "s3_key": result['s3_key'],
            "logo_filename": result['logo_filename'],
            "result": success_response.model_dump()
        }

    except ValueError as ve:
        # Handle validation errors from service (e.g., invalid file type)
        logger.warning(f"Validation error in upload_company_logo endpoint: {ve}")
        error_response = create_error_response(str(ve))
        raise HTTPException(
            status_code=400,
            detail={"result": error_response.model_dump()}
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in upload_company_logo endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )