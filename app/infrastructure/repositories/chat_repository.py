"""Repository for chat database operations using actual CHATS table structure."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class ChatRepository:
    """
    Repository for CHATS table operations.

    Table structure:
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

    @retry_on_db_error(max_retries=3, delay=1)
    def create_chat(
        self,
        id_usuario: int,
        id_area: int,
        id_empresa: int,
        titulo: str
    ) -> Optional[int]:
        """
        Create a new chat using stored procedure SP_CREATE_CHAT

        Args:
            id_usuario: User ID
            id_area: Area identifier
            id_empresa: Company identifier
            titulo: Chat title

        Returns:
            ID_CHAT of the created chat, or None if creation failed
        """
        try:
            query = text("""
                EXEC SP_CREATE_CHAT
                @ID_USUARIO = :id_usuario,
                @ID_AREA = :id_area,
                @ID_EMPRESA = :id_empresa,
                @TITULO = :titulo
            """)

            result = self.db.execute(query, {
                'id_usuario': id_usuario,
                'id_area': id_area,
                'id_empresa': id_empresa,
                'titulo': titulo
            })

            chat_data = result.fetchone()
            result.close()

            if not chat_data:
                logger.warning(f"No response from SP_CREATE_CHAT")
                return None

            chat_dict = dict(chat_data._mapping) if hasattr(chat_data, '_mapping') else dict(zip(result.keys(), chat_data))
            chat_id = chat_dict.get('ID_CHAT')

            if chat_id:
                self.db.commit()
                return chat_id

            return None

        except Exception as e:
            logger.error(f"Error creating chat with SP: {e}")
            self.db.rollback()
            return None

    @retry_on_db_error(max_retries=3, delay=1)
    def get_chats_by_user(self, user_id: int) -> List[Dict[str, Any]]:
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

    @retry_on_db_error(max_retries=3, delay=1)
    def update_chat_titulo(
        self,
        chat_id: int,
        titulo: str
    ) -> bool:
        """
        Update chat TITULO using stored procedure SP_UPDATE_CHAT_TITULO

        Args:
            chat_id: Chat identifier
            titulo: New chat title

        Returns:
            True if successful, False otherwise
        """
        try:
            query = text("""
                EXEC SP_UPDATE_CHAT_TITULO
                    @ID_CHAT = :id_chat,
                    @TITULO = :titulo
            """)

            self.db.execute(query, {
                'id_chat': chat_id,
                'titulo': titulo
            })
            self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating chat titulo {chat_id}: {e}")
            self.db.rollback()
            return False

    @retry_on_db_error(max_retries=3, delay=1)
    def delete_chat(self, chat_id: int) -> bool:
        """
        Soft delete chat using stored procedure SP_UPDATE_CHAT_ESTADO_REGISTRO
        Sets ID_ESTADO_REGISTRO to 0 (deleted/inactive)

        Args:
            chat_id: Chat identifier

        Returns:
            True if deleted successfully, False otherwise
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
            return True

        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            self.db.rollback()
            return False

    @retry_on_db_error(max_retries=3, delay=1)
    def update_ultimo_mensaje_fecha(self, chat_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA using stored procedure SP_UPDATE_CHAT_ULTIMO_MENSAJE_FECHA

        Args:
            chat_id: Chat identifier

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            query = text("""
                EXEC SP_UPDATE_CHAT_ULTIMO_MENSAJE_FECHA
                @ID_CHAT = :id_chat
            """)

            self.db.execute(query, {'id_chat': chat_id})
            self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating ultimo_mensaje_fecha for chat {chat_id}: {e}")
            self.db.rollback()
            return False
