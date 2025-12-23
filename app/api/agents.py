"""API endpoints for agents management."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.agents_service import AgentsService
from app.models.response_models import create_success_response, create_error_response
from app.models.agent_models import CreateAgentRequest
from app.utils.jwt_auth import get_current_user_with_company_validation, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_agents_service() -> AgentsService:
    """Get singleton AgentsService from container."""
    return container.get_agents_service()


@router.get("/get_agentes/{id_empresa}")
async def get_agentes_endpoint(
    id_empresa: int,
    agents_service: AgentsService = Depends(get_agents_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Get all agents for a company endpoint.

    Fetches all agents for a specific company using SP_AGENTES_LST.
    Requires JWT authentication and validates company access.

    Args:
        id_empresa: Company ID from URL path

    Returns:
        Dict with:
        - agentes: List of agent dictionaries
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

        # Get agentes using service
        agentes = await agents_service.get_agentes(
            db=db,
            id_empresa=id_empresa
        )

        if agentes is None:
            error_response = create_error_response("Error al obtener los agentes")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Agentes obtenidos exitosamente")
        return {
            "agentes": agentes,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_agentes endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.post("/create_agente")
async def create_agente_endpoint(
    http_request: Request,
    request: CreateAgentRequest,
    agents_service: AgentsService = Depends(get_agents_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """
    Create a new agent endpoint.

    Creates a new agent using SP_CREATE_AGENTE.
    Requires JWT authentication and validates company access.

    Args:
        request: CreateAgentRequest with agent details and areas
            - numero_telf: Phone number
            - id_tipo_agente: Agent type ID
            - id_empresa: Company ID
            - acceso_general: General access flag (0 or 1)
            - areas_string: Comma-separated area IDs

    Returns:
        Dict with agent creation results including:
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

        # Create agent using service (user_id is the current user creating the new agent)
        results = await agents_service.create_agente(
            db=db,
            id_usuario=user_id,
            numero_telf=request.numero_telf,
            codigo_pais=request.codigo_pais,
            id_tipo_agente=request.id_tipo_agente,
            id_empresa=request.id_empresa,
            acceso_general=request.acceso_general,
            areas_string=request.areas_string
        )

        # SP may not return results for now, so we just treat empty results as success
        # Check if the stored procedure returned an error message
        if results and 'ID_TIPO_MENSAJE' in results[0]:
            tipo_mensaje = results[0].get('ID_TIPO_MENSAJE')
            mensaje = results[0].get('MENSAJE', 'Error desconocido')

            # Log when ID_TIPO_MENSAJE is not 2 (success)
            if tipo_mensaje != 2:
                logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje} in create_agente: {mensaje}")

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

        success_response = create_success_response("Agente creado exitosamente")
        return {
            "results": results if results else [],
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_agente endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )