"""Chat models for managing chat sessions."""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime


# =============================================
# Pydantic Request/Response Models for CHATS table
# =============================================

class ChatCreateRequest(BaseModel):
    """Request to create a new chat"""
    titulo: Optional[str] = None  # If None, auto-generate from first message
    id_empresa: int
    id_area: int


class ChatUpdateRequest(BaseModel):
    """Request to update chat TITULO"""
    titulo: str


class ChatResponse(BaseModel):
    """Response model for chat details - matches CHATS table structure"""
    model_config = ConfigDict(from_attributes=True)

    id_chat: int
    id_area: int
    id_empresa: int
    titulo: str
    ultimo_mensaje_fecha: Optional[datetime]
    usucre: Optional[str]
    usumod: Optional[str]
    fchmod: Optional[datetime]
    fchcre: datetime
    id_estado_registro: int


class ChatListItem(BaseModel):
    """Lightweight chat item for list view"""
    model_config = ConfigDict(from_attributes=True)

    id_chat: int
    titulo: str
    ultimo_mensaje_fecha: Optional[datetime]
    fchcre: datetime
    id_area: int
    id_empresa: int
    id_estado_registro: int


class ChatListResponse(BaseModel):
    """Response for chat list endpoint"""
    chats: List[ChatListItem]
    total_count: int
    page: int
    page_size: int
