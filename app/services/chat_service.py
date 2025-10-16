"""Service for managing chat sessions following hexagonal architecture."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models.chat_models import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse,
    ChatListItem
)

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

    async def create_chat(
        self,
        request: ChatCreateRequest,
        username: str
    ) -> Optional[ChatResponse]:
        """
        Create a new chat session

        Args:
            request: Chat creation request with id_empresa, id_area, and optional titulo
            username: Username of the user creating the chat (for USUCRE)

        Returns:
            ChatResponse with created chat data, or None if creation failed
        """
        try:
            # Auto-generate title if not provided
            titulo = request.titulo or await self._generate_auto_title()

            query = text("""
                INSERT INTO dbo.CHATS (ID_AREA, ID_EMPRESA, TITULO, USUCRE, FCHCRE, ID_ESTADO_REGISTRO)
                OUTPUT INSERTED.*
                VALUES (:id_area, :id_empresa, :titulo, :usucre, GETDATE(), 1)
            """)

            result = self.db.execute(query, {
                'id_area': request.id_area,
                'id_empresa': request.id_empresa,
                'titulo': titulo,
                'usucre': username
            })

            chat_data = result.fetchone()
            self.db.commit()

            if chat_data:
                return self._map_to_chat_response(dict(chat_data._mapping))
            return None

        except Exception as e:
            logger.error(f"Error in create_chat service: {e}")
            self.db.rollback()
            raise

    async def get_chat(self, chat_id: int) -> Optional[ChatResponse]:
        """
        Get chat details by ID

        Args:
            chat_id: Chat identifier

        Returns:
            ChatResponse with chat data, or None if not found
        """
        try:
            query = text("""
                SELECT
                    ID_CHAT,
                    ID_AREA,
                    ID_EMPRESA,
                    TITULO,
                    ULTIMO_MENSAJE_FECHA,
                    USUCRE,
                    USUMOD,
                    FCHMOD,
                    FCHCRE,
                    ID_ESTADO_REGISTRO
                FROM dbo.CHATS
                WHERE ID_CHAT = :chat_id
                    AND ID_ESTADO_REGISTRO = 1
            """)

            result = self.db.execute(query, {'chat_id': chat_id})
            chat_data = result.fetchone()

            if chat_data:
                return self._map_to_chat_response(dict(chat_data._mapping))
            return None

        except Exception as e:
            logger.error(f"Error in get_chat service: {e}")
            return None

    async def get_chats_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List chats for a specific user using SP_GET_USER_CHATS

        Args:
            user_id: User identifier

        Returns:
            List of dictionaries with chat data
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_GET_USER_CHATS @ID_USUARIO = ?", user_id)

                chats = []

                # Check if we have results
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Convert rows to list of dictionaries
                    for row in rows:
                        chat_dict = dict(zip(columns, row))
                        chats.append(chat_dict)

                cursor.close()
                return chats

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_chats_by_user: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error listing chats for user {user_id}: {e}")
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
            query = text("""
                EXEC SP_UPDATE_CHAT_TITULO
                    @ID_CHAT = :id_chat,
                    @TITULO = :titulo
            """)

            self.db.execute(query, {
                'id_chat': chat_id,
                'titulo': request.titulo
            })
            self.db.commit()

            return {
                "ID_TIPO_MENSAJE": 2,
                "MENSAJE": "Chat title updated successfully"
            }

        except Exception as e:
            logger.error(f"Error in update_chat_titulo service: {e}")
            self.db.rollback()
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
            query = text("""
                EXEC SP_UPDATE_CHAT_ESTADO_REGISTRO
                    @ID_CHAT = :id_chat,
                    @ID_ESTADO_REGISTRO = :id_estado_registro
            """)

            self.db.execute(query, {
                'id_chat': chat_id,
                'id_estado_registro': 0  # 0 = deleted/inactive
            })

            self.db.commit()

            return {
                "ID_TIPO_MENSAJE": 2,
                "MENSAJE": "Chat deleted successfully"
            }

        except Exception as e:
            logger.error(f"Error in delete_chat service: {e}")
            self.db.rollback()
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
            query = text("""
                UPDATE dbo.CHATS
                SET ULTIMO_MENSAJE_FECHA = GETDATE()
                WHERE ID_CHAT = :chat_id
                    AND ID_ESTADO_REGISTRO = 1
            """)

            result = self.db.execute(query, {'chat_id': chat_id})
            self.db.commit()
            return result.rowcount > 0

        except Exception as e:
            logger.error(f"Error in update_ultimo_mensaje_fecha: {e}")
            self.db.rollback()
            return False

    async def generate_chat_title_from_message(self, message: str) -> str:
        """
        Generate a concise chat title from first message
        Simple implementation: truncate to 50 characters

        Args:
            message: User's first message

        Returns:
            Generated title string
        """
        try:
            max_length = 50
            if len(message) <= max_length:
                return message
            return message[:max_length].rsplit(' ', 1)[0] + "..."

        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Nueva Conversación"

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
