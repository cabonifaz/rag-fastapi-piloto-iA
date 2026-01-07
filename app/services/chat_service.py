"""Service for managing chat sessions following hexagonal architecture."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import logging

from app.models.chat_models import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse,
    ChatListItem
)
from app.infrastructure.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for chat operations.
    Handles business logic for creating, retrieving, updating, and deleting chats.

    Database table structure (CHATS):
    - ID_CHAT: Primary key (auto-generated)
    - ID_AREA: Area identifier
    - ID_EMPRESA: Company identifier
    - TITULO: Chat title
    - ULTIMO_MENSAJE_FECHA: Last message timestamp
    - USUCRE: User who created the chat
    - USUMOD: User who last modified the chat
    - FCHMOD: Last modification date
    - FCHCRE: Creation date (auto-generated)
    - ID_ESTADO_REGISTRO: Status (1=active, 0=deleted)
    """

    def __init__(self):
        """Initialize stateless ChatService - no db parameter."""
        pass

    async def get_chats_by_user(self, db: Session, user_id: int) -> List[Dict[str, Any]]:
        """
        List chats for a specific user using SP_GET_USER_CHATS

        Args:
            db: Database session
            user_id: User identifier

        Returns:
            List of dictionaries with chat data
        """
        try:
            repository = ChatRepository(db)
            return repository.get_chats_by_user(user_id)
        except Exception as e:
            logger.error(f"Error in get_chats_by_user service: {e}")
            return []

    async def update_chat_titulo(
        self,
        db: Session,
        chat_id: int,
        request: ChatUpdateRequest,
    ) -> Dict[str, Any]:
        """
        Update chat TITULO using stored procedure SP_UPDATE_CHAT_TITULO

        Args:
            db: Database session
            chat_id: Chat identifier
            request: Update request with new titulo

        Returns:
            Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE
        """
        try:
            repository = ChatRepository(db)
            success = repository.update_chat_titulo(
                chat_id=chat_id,
                titulo=request.titulo
            )

            if success:
                return {
                    "ID_TIPO_MENSAJE": 2,
                    "MENSAJE": "Chat title updated successfully"
                }
            else:
                return {
                    "ID_TIPO_MENSAJE": 1,
                    "MENSAJE": "Failed to update chat title"
                }

        except Exception as e:
            logger.error(f"Error in update_chat_titulo service: {e}")
            return {
                "ID_TIPO_MENSAJE": 1,
                "MENSAJE": f"Failed to update chat title: {str(e)}"
            }

    async def delete_chat(self, db: Session, chat_id: int) -> Dict[str, Any]:
        """
        Soft delete a chat using stored procedure SP_UPDATE_CHAT_ESTADO_REGISTRO
        Sets ID_ESTADO_REGISTRO to 0 (deleted/inactive)

        Args:
            db: Database session
            chat_id: Chat identifier

        Returns:
            Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE
        """
        try:
            repository = ChatRepository(db)
            success = repository.delete_chat(chat_id)

            if success:
                return {
                    "ID_TIPO_MENSAJE": 2,
                    "MENSAJE": "Chat deleted successfully"
                }
            else:
                return {
                    "ID_TIPO_MENSAJE": 1,
                    "MENSAJE": "Failed to delete chat"
                }

        except Exception as e:
            logger.error(f"Error in delete_chat service: {e}")
            return {
                "ID_TIPO_MENSAJE": 1,
                "MENSAJE": f"Failed to delete chat: {str(e)}"
            }

    async def update_ultimo_mensaje_fecha(self, db: Session, chat_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA to current timestamp
        Should be called when a new message is added to the chat

        Args:
            db: Database session
            chat_id: Chat identifier

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            repository = ChatRepository(db)
            return repository.update_ultimo_mensaje_fecha(chat_id)
        except Exception as e:
            logger.error(f"Error in update_ultimo_mensaje_fecha service: {e}")
            return False

    # =============================================
    # Helper Methods
    # =============================================

    async def _generate_auto_title(self) -> str:
        """Generate default title for new chat"""
        from datetime import datetime
        return f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def _map_to_chat_response(self, chat_data: Dict[str, Any]) -> ChatResponse:
        """Map database result to ChatResponse"""
        return ChatResponse(
            id_chat=chat_data['ID_CHAT'],
            id_area=chat_data['ID_AREA'],
            id_empresa=chat_data['ID_EMPRESA'],
            titulo=chat_data['TITULO'],
            ultimo_mensaje_fecha=chat_data.get('ULTIMO_MENSAJE_FECHA'),
            usucre=chat_data.get('USUCRE'),
            usumod=chat_data.get('USUMOD'),
            fchmod=chat_data.get('FCHMOD'),
            fchcre=chat_data['FCHCRE'],
            id_estado_registro=chat_data['ID_ESTADO_REGISTRO']
        )

    def _map_to_chat_list_item(self, chat_data: Dict[str, Any]) -> ChatListItem:
        """Map database result to ChatListItem"""
        return ChatListItem(
            id_chat=chat_data['ID_CHAT'],
            titulo=chat_data['TITULO'],
            ultimo_mensaje_fecha=chat_data.get('ULTIMO_MENSAJE_FECHA'),
            fchcre=chat_data['FCHCRE'],
            id_area=chat_data['ID_AREA'],
            id_empresa=chat_data['ID_EMPRESA'],
            id_estado_registro=chat_data['ID_ESTADO_REGISTRO']
        )
