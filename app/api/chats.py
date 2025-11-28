"""API endpoints for chat management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.chat_service import ChatService
from app.models.chat_models import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse
)
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    """Get singleton ChatService from container."""
    return container.get_chat_service()


@router.get("/get_chats")
async def get_user_chats(
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get user chats endpoint

    Retrieves all chats for the authenticated user using their user ID from JWT.

    Returns:
        List of chats for the user

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')

        if not user_id:
            error_response = create_error_response("Información de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Get user chats from service
        chats = await chat_service.get_chats_by_user(db, user_id)

        if chats is None:
            error_response = create_error_response("Error al obtener los chats del usuario")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Chats obtenidos exitosamente")
        return {
            "chats": chats,  # Can be [] for new users
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get user chats endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.patch("/{chat_id}")
async def update_chat_titulo_endpoint(
    chat_id: int,
    request: ChatUpdateRequest,
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update chat title (TITULO)

    Requires JWT authentication. Updates the chat's TITULO field using SP_UPDATE_CHAT_TITULO.

    Args:
        chat_id: Chat identifier
        request: ChatUpdateRequest with new titulo

    Returns:
        Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        result = await chat_service.update_chat_titulo(
            db=db,
            chat_id=chat_id,
            request=request
        )

        # Check if the operation failed
        tipo_mensaje = result.get("ID_TIPO_MENSAJE")

        # Log when ID_TIPO_MENSAJE is not 2 (success)
        if tipo_mensaje and tipo_mensaje != 2:
            logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje}: {result.get('MENSAJE')}")

        if tipo_mensaje == 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )
        elif tipo_mensaje == 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_chat_titulo endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "ID_TIPO_MENSAJE": 1,
                "MENSAJE": "Error interno del servidor"
            }
        )


@router.delete("/{chat_id}")
async def delete_chat_endpoint(
    chat_id: int,
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a chat (soft delete)

    Requires JWT authentication. Soft deletes the chat by setting ID_ESTADO_REGISTRO to 0
    using SP_UPDATE_CHAT_ESTADO_REGISTRO. The chat data is preserved but marked as inactive.

    Args:
        chat_id: Chat identifier

    Returns:
        Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        result = await chat_service.delete_chat(db=db, chat_id=chat_id)

        # Check if the operation failed
        tipo_mensaje = result.get("ID_TIPO_MENSAJE")

        # Log when ID_TIPO_MENSAJE is not 2 (success)
        if tipo_mensaje and tipo_mensaje != 2:
            logger.warning(f"SP returned ID_TIPO_MENSAJE={tipo_mensaje}: {result.get('MENSAJE')}")

        if tipo_mensaje == 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )
        elif tipo_mensaje == 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in delete_chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "ID_TIPO_MENSAJE": 1,
                "MENSAJE": "Error interno del servidor"
            }
        )
