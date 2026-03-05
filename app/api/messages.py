"""API endpoints for message management."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
import json
from typing import Optional, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.container import container

from app.services.message_service import MessageService
from app.models.message_models import (
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    GetMessagesByChat,
    GenerateAttachmentPresignedUrlsRequest,
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
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum messages to return (if omitted, loaded from DB parameter)") ,
    last_evaluated_key: Optional[str] = Query(None, description="Pagination key (JSON string)"),
    db: Session = Depends(get_db),
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
            last_evaluated_key=last_key,
            db=db
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


@router.post("/attachment-upload-urls")
async def generate_attachment_presigned_urls_endpoint(
    request: GenerateAttachmentPresignedUrlsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation),
    message_service: MessageService = Depends(get_message_service),
):
    """
    Generate presigned PUT URLs for direct S3 upload of chat attachments.

    Request Body Parameters:
        - filenames: List of filenames to upload
        - timestamp: Timestamp string used as folder prefix in S3 key

    Returns:
        List of objects with presigned_url, s3_key, and filename
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        blob_storage = container.get_blob_storage()

        uploads = await message_service.generate_attachment_presigned_urls(
            blob_storage=blob_storage,
            user_id=user_id,
            timestamp=request.timestamp,
            filenames=request.filenames,
        )

        return {"uploads": uploads}

    except Exception as e:
        logger.error(f"Unexpected error in generate_attachment_presigned_urls endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )