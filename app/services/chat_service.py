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

    def __init__(self, db: Session):
        self.db = db
        self.repository = ChatRepository(db)

    async def create_chat(
        self,
        request: ChatCreateRequest,
        user_id: int
    ) -> Optional[ChatResponse]:
        """
        Create a new chat session using stored procedure

        Args:
            request: Chat creation request with id_empresa, id_area, and optional titulo
            user_id: User ID creating the chat

        Returns:
            ChatResponse with created chat data, or None if creation failed
        """
        try:
            # Auto-generate title if not provided
            titulo = request.titulo or await self._generate_auto_title()

            # Use repository to create chat with SP_CREATE_CHAT
            chat_id = self.repository.create_chat(
                id_usuario=user_id,
                id_area=request.id_area,
                id_empresa=request.id_empresa,
                titulo=titulo
            )

            if not chat_id:
                return None

            # Fetch the created chat to return full response
            chat_data = self.repository.get_chat_by_id(chat_id)
            if chat_data:
                return self._map_to_chat_response(chat_data)
            return None

        except Exception as e:
            logger.error(f"Error in create_chat service: {e}")
            raise

    async def get_chats_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List chats for a specific user using SP_GET_USER_CHATS

        Args:
            user_id: User identifier

        Returns:
            List of dictionaries with chat data
        """
        try:
            return self.repository.get_chats_by_user(user_id)
        except Exception as e:
            logger.error(f"Error in get_chats_by_user service: {e}")
            return []

    async def update_chat_titulo(
        self,
        chat_id: int,
        request: ChatUpdateRequest,
    ) -> Dict[str, Any]:
        """
        Update chat TITULO using stored procedure SP_UPDATE_CHAT_TITULO

        Args:
            chat_id: Chat identifier
            request: Update request with new titulo

        Returns:
            Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE
        """
        try:
            success = self.repository.update_chat_titulo(
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

    async def delete_chat(self, chat_id: int) -> Dict[str, Any]:
        """
        Soft delete a chat using stored procedure SP_UPDATE_CHAT_ESTADO_REGISTRO
        Sets ID_ESTADO_REGISTRO to 0 (deleted/inactive)

        Args:
            chat_id: Chat identifier

        Returns:
            Dict with ID_TIPO_MENSAJE (2=success, 1=failure) and MENSAJE
        """
        try:
            success = self.repository.delete_chat(chat_id)

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

    async def update_ultimo_mensaje_fecha(self, chat_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA to current timestamp
        Should be called when a new message is added to the chat

        Args:
            chat_id: Chat identifier

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            return self.repository.update_ultimo_mensaje_fecha(chat_id)
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
