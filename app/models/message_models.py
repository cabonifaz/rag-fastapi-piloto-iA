"""Message models for DynamoDB chat messages."""

from pydantic import BaseModel, Field
from typing import Optional, List


# =============================================
# DynamoDB Message Models
# =============================================

class MessageCreate(BaseModel):
    """Request to create a new message"""
    chat_id: str
    created_at: str  # Timestamp as string (milliseconds since epoch)
    sender: int  # 0 = user, 1 = assistant
    message: str


class MessageResponse(BaseModel):
    """Simplified response model for a message, matching frontend expectations."""
    id: str          # Unique identifier, populated from created_at
    chat_id: str
    created_at: str
    sender: int
    message: str
    attachment_keys: Optional[List[str]] = None


class MessageListResponse(BaseModel):
    """Response for listing messages"""
    messages: List[MessageResponse]
    total_count: int
    last_evaluated_key: Optional[dict] = None  # For pagination


class MessageUpdate(BaseModel):
    """Request to update a message (soft delete)"""
    id_estado_registro: int  # Set to 0 for soft delete


class GetMessagesByChat(BaseModel):
    """Request to get messages for a specific chat"""
    chat_id: str
    company_id: int
    area_id: int


class GenerateAttachmentPresignedUrlsRequest(BaseModel):
    """Request to generate presigned PUT URLs for chat attachment uploads"""
    company_id: int
    area_id: int
    filenames: List[str]
    timestamp: str
