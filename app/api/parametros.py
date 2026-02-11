"""API endpoints for parametros management."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.parametros_service import ParametrosService
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_parametros_service() -> ParametrosService:
    """Get singleton ParametrosService from container."""
    return container.get_parametros_service()


@router.get("/get_param_by_id_maestro")
async def get_param_by_id_maestro_endpoint(
    grp_id_maestro: str = Query(..., min_length=1, max_length=50, description="Group ID (GRP_ID_MAESTRO) to look up"),
    parametros_service: ParametrosService = Depends(get_parametros_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all parameter rows by GRP_ID_MAESTRO.

    Fetches all parameter rows for the given GRP_ID_MAESTRO using SP_PARAMETROS_LST.
    Requires JWT authentication.

    Query Parameters:
        grp_id_maestro: Group ID (GRP_ID_MAESTRO) to look up

    Returns:
        Dict with:
        - data: List of parameter dictionaries
        - result: Success/error response

    Raises:
        HTTPException: 401 for auth errors, 404 if not found, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        results = await parametros_service.get_param_by_id_maestro(db, grp_id_maestro)

        if not results:
            error_response = create_error_response(f"Parametros no encontrados para grp_id_maestro={grp_id_maestro}")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Parametros obtenidos exitosamente")
        return {
            "data": results,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_param_by_id_maestro endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
