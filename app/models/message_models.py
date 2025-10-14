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
    """Response model for a message"""
    chat_id: str
    created_at: str  # Timestamp as string (milliseconds since epoch)
    id_estado_registro: int  # 1 = active, 0 = deleted
    sender: int  # 0 = user, 1 = assistant
    message: str

    # Composite key for DynamoDB
    chat_id_estado: str = Field(alias="chat_id#id_estado_registro")

    class Config:
        populate_by_name = True


class MessageListResponse(BaseModel):
    """Response for listing messages"""
    messages: List[MessageResponse]
    total_count: int
    last_evaluated_key: Optional[dict] = None  # For pagination


class MessageUpdate(BaseModel):
    """Request to update a message (soft delete)"""
    id_estado_registro: int  # Set to 0 for soft delete
