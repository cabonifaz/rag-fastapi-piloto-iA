"""API endpoints for phone code management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.phone_code_service import PhoneCodeService
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_phone_code_service() -> PhoneCodeService:
    """Get singleton PhoneCodeService from container."""
    return container.get_phone_code_service()


@router.get("/get_phone_codes")
async def get_phone_codes_endpoint(
    phone_code_service: PhoneCodeService = Depends(get_phone_code_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all phone codes endpoint.

    Fetches all country phone codes using SP_CODIGOS_PAIS_LST.
    Requires JWT authentication.

    Returns:
        Dict with:
        - phone_codes: List of phone code dictionaries with:
            - CODIGO_NUMERICO: Numeric country code
            - CODIGO_ISO: ISO country code (e.g., PE, ES, MX)
            - NOMBRE_PAIS: Country name
            - PREFIJO_TELEFONICO: Phone prefix (e.g., +51, +34, +52)
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 500 for server errors
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
        if role_id not in [1, 2, 3]:
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Permisos insuficientes"}}
            )

        # Get phone codes using service
        phone_codes = await phone_code_service.get_phone_codes(db=db)

        if phone_codes is None:
            error_response = create_error_response("Error al obtener los códigos de país")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Códigos de país obtenidos exitosamente")
        return {
            "phone_codes": phone_codes,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_phone_codes endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
