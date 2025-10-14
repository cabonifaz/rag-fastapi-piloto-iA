"""API endpoints for message management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, Dict, Any
import logging

from app.services.message_service import MessageService
from app.models.message_models import (
    MessageCreate,
    MessageResponse,
    MessageListResponse
)
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_message_service() -> MessageService:
    """Dependency injection for MessageService"""
    return MessageService()


@router.post("", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message_endpoint(
    request: MessageCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    message_service: MessageService = Depends(get_message_service)
):
    """
    Create a new message in DynamoDB

    Requires JWT authentication. Creates a message with the provided chat_id, created_at, sender, and message.

    Args:
        request: MessageCreate with chat_id, created_at (timestamp as string), sender (0=user, 1=assistant), message

    Returns:
        MessageResponse with created message details

    Raises:
        HTTPException: 500 if message creation fails
    """
    try:
        message = await message_service.create_message(request=request)

        if not message:
            error_response = create_error_response("Error al crear el mensaje")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"result": error_response.model_dump()}
            )

        return message

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_message endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.get("/chat/{chat_id}", response_model=MessageListResponse)
async def get_messages_by_chat_endpoint(
    chat_id: str,
    limit: int = Query(50, ge=1, le=100, description="Maximum messages to return"),
    last_evaluated_key: Optional[str] = Query(None, description="Pagination key (JSON string)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    message_service: MessageService = Depends(get_message_service)
):
    """
    Get messages for a specific chat with pagination

    Requires JWT authentication. Returns messages ordered from oldest to newest.

    Query Parameters:
        - chat_id: Chat identifier (path parameter)
        - limit: Maximum messages to return (default: 50, min: 1, max: 100)
        - last_evaluated_key: For pagination, pass the last_evaluated_key from previous response

    Returns:
        MessageListResponse with messages list and pagination info

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        # Parse last_evaluated_key if provided
        last_key = None
        if last_evaluated_key:
            try:
                import json
                last_key = json.loads(last_evaluated_key)
            except:
                error_response = create_error_response("Invalid last_evaluated_key format")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"result": error_response.model_dump()}
                )

        message_list = await message_service.get_messages_by_chat(
            chat_id=chat_id,
            limit=limit,
            last_evaluated_key=last_key
        )

        return message_list

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_messages_by_chat endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )


@router.delete("/{chat_id}/{created_at}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message_endpoint(
    chat_id: str,
    created_at: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    message_service: MessageService = Depends(get_message_service)
):
    """
    Soft delete a message

    Requires JWT authentication. Soft deletes the message by setting id_estado_registro to 0.
    The message data is preserved but marked as inactive.

    Args:
        chat_id: Chat identifier
        created_at: Message timestamp (milliseconds since epoch as string)

    Returns:
        204 No Content on successful deletion

    Raises:
        HTTPException: 404 if message not found, 500 for server errors
    """
    try:
        success = await message_service.soft_delete_message(
            chat_id=chat_id,
            created_at=created_at
        )

        if not success:
            error_response = create_error_response("Mensaje no encontrado")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"result": error_response.model_dump()}
            )

        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in delete_message endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )
