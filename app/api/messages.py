"""API endpoints for message management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
import json
from typing import Optional, Dict, Any
import logging

from app.services.message_service import MessageService
from app.models.message_models import (
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    GetMessagesByChat
)
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user_with_company_area_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def get_message_service() -> MessageService:
    """Dependency injection for MessageService"""
    return MessageService()


@router.post("/chat", response_model=MessageListResponse)
async def get_messages_by_chat_endpoint(
    request: GetMessagesByChat,
    limit: int = Query(50, ge=1, le=100, description="Maximum messages to return"),
    last_evaluated_key: Optional[str] = Query(None, description="Pagination key (JSON string)"),
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation),
    message_service: MessageService = Depends(get_message_service)
):
    """
    Get messages for a specific chat with pagination

    Requires JWT authentication. Returns messages ordered from oldest to newest.

    Request Body Parameters:
        - chat_id: Chat identifier
        - company_id: Company identifier
        - area_id: Area identifier

    Query Parameters:
        - limit: Maximum messages to return (default: 50, min: 1, max: 100)
        - last_evaluated_key: For pagination, pass the last_evaluated_key from previous response

    Returns:
        MessageListResponse with messages list and pagination info

    Raises:
        HTTPException: 400 for invalid parameters, 500 for server errors
    """
    try:
        # Parse last_evaluated_key if provided
        last_key = None
        if last_evaluated_key:
            try:
                last_key = json.loads(last_evaluated_key)
            except:
                error_response = create_error_response("Invalid last_evaluated_key format")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"result": error_response.model_dump()}
                )

        message_list = await message_service.get_messages_by_chat(
            chat_id=request.chat_id,
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