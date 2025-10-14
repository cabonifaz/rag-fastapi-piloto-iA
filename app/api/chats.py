"""API endpoints for chat management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import logging

from app.core.database import get_db
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


def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    """Dependency injection for ChatService"""
    return ChatService(db)


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_endpoint(
    request: ChatCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Create a new chat session

    Requires JWT authentication. Creates a new chat with the provided titulo, id_empresa, and id_area.
    If titulo is not provided, an auto-generated title will be used.

    Args:
        request: ChatCreateRequest with titulo (optional), id_empresa, id_area

    Returns:
        ChatResponse with created chat details

    Raises:
        HTTPException: 500 if chat creation fails
    """
    try:
        username = current_user.get('USUARIO', 'unknown')

        chat = await chat_service.create_chat(request=request, username=username)

        if not chat:
            error_response = create_error_response("Error al crear el chat")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"result": error_response.model_dump()}
            )

        return chat

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_chat endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.get("", response_model=ChatListResponse)
async def list_chats_endpoint(
    id_empresa: Optional[int] = Query(None, description="Filter by company ID"),
    id_area: Optional[int] = Query(None, description="Filter by area ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    List user's chats with optional filtering

    Requires JWT authentication. Returns paginated list of active chats.
    Can be filtered by id_empresa and/or id_area.

    Query Parameters:
        - id_empresa: Filter by company (optional)
        - id_area: Filter by area (optional)
        - page: Page number (default: 1, min: 1)
        - page_size: Items per page (default: 50, min: 1, max: 100)

    Returns:
        ChatListResponse with paginated chat list and metadata

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        chat_list = await chat_service.list_chats(
            id_empresa=id_empresa,
            id_area=id_area,
            page=page,
            page_size=page_size
        )

        return chat_list

    except Exception as e:
        logger.error(f"Unexpected error in list_chats endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat_endpoint(
    chat_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Get chat details by ID

    Requires JWT authentication. Returns full chat information.

    Args:
        chat_id: Chat identifier

    Returns:
        ChatResponse with chat details

    Raises:
        HTTPException: 404 if chat not found, 500 for server errors
    """
    try:
        chat = await chat_service.get_chat(chat_id=chat_id)

        if not chat:
            error_response = create_error_response("Chat no encontrado")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"result": error_response.model_dump()}
            )

        return chat

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_chat endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat_titulo_endpoint(
    chat_id: int,
    request: ChatUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Update chat title (TITULO)

    Requires JWT authentication. Updates the chat's TITULO field.

    Args:
        chat_id: Chat identifier
        request: ChatUpdateRequest with new titulo

    Returns:
        ChatResponse with updated chat details

    Raises:
        HTTPException: 404 if chat not found, 500 for server errors
    """
    try:
        username = current_user.get('USUARIO', 'unknown')

        chat = await chat_service.update_chat_titulo(
            chat_id=chat_id,
            request=request,
            username=username
        )

        if not chat:
            error_response = create_error_response("Chat no encontrado")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"result": error_response.model_dump()}
            )

        return chat

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in update_chat_titulo endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_endpoint(
    chat_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Delete a chat (soft delete)

    Requires JWT authentication. Soft deletes the chat by setting ID_ESTADO_REGISTRO to 0.
    The chat data is preserved but marked as inactive.

    Args:
        chat_id: Chat identifier

    Returns:
        204 No Content on successful deletion

    Raises:
        HTTPException: 404 if chat not found, 500 for server errors
    """
    try:
        username = current_user.get('USUARIO', 'unknown')

        success = await chat_service.delete_chat(chat_id=chat_id, username=username)

        if not success:
            error_response = create_error_response("Chat no encontrado")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"result": error_response.model_dump()}
            )

        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in delete_chat endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )
