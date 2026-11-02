import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.container import container
from app.core.database import get_db
from app.models.mcp_chat_models import (
    MCPAvailabilityRequest,
    MCPAvailabilityResponse,
    MCPChatRequest,
    MCPChatResponse,
)
from app.services.mcp_chat_service import MCPChatService
from app.utils.jwt_auth import get_current_user_with_company_area_validation

logger = logging.getLogger(__name__)
router = APIRouter()


def get_mcp_chat_service() -> MCPChatService:
    return container.get_mcp_chat_service()


@router.post("/availability", response_model=MCPAvailabilityResponse)
async def mcp_availability_endpoint(
    request: MCPAvailabilityRequest,
    service: MCPChatService = Depends(get_mcp_chat_service),
    db: Session = Depends(get_db),
    _current_user: Dict[str, Any] = Depends(
        get_current_user_with_company_area_validation
    ),
):
    """Return whether MCP is enabled for the requested company and area."""
    return {"enabled": service.is_available(db, request.company_id, request.area_id)}


@router.post("/chat", response_model=MCPChatResponse)
async def mcp_chat_endpoint(
    request: MCPChatRequest,
    service: MCPChatService = Depends(get_mcp_chat_service),
    db: Session = Depends(get_db),
):
    """Generate a Bedrock response using MCP tools, without RAG."""
    try:
        return await service.process(
            message=request.message,
            company_id=request.company_id,
            area_id=request.area_id,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"result": {"idTipoMensaje": 1, "mensaje": str(exc)}},
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"result": {"idTipoMensaje": 1, "mensaje": str(exc)}},
        ) from exc
    except ConnectionError as exc:
        logger.error("MCP chat connection error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"result": {"idTipoMensaje": 1, "mensaje": str(exc)}},
        ) from exc
