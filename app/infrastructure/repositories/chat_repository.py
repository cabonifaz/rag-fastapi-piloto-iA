"""Repository for chat database operations using actual CHATS table structure."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging

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

    # =============================================
    # Chat Operations
    # =============================================

    def create_chat(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        titulo: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new chat using stored procedure

        Args:
            id_empresa: Company identifier
            id_area: Area identifier  
            titulo: Chat title
            id_usuario: User ID 

        Returns:
            ID_CHAT of the created chat, or None if creation failed
        """
        try:
            # TODO: Replace with actual stored procedure name when available
            # query = text("EXEC dbo.SP_CHAT_CREATE @ID_AREA = :id_area, @ID_EMPRESA = :id_empresa, @TITULO = :titulo, @USUCRE = :usucre")

            # Temporary direct INSERT until SP is created
            query = text("EXEC SP_CREATE_CHAT @ID_USUARIO = :id_usuario, @ID_AREA = :id_area, @ID_EMPRESA = :id_empresa, @TITULO = :titulo")

            result = self.db.execute(query, {
                'id_usuario': id_usuario,
                'id_area': id_area,
                'id_empresa': id_empresa,
                'titulo': titulo
            })

            chat_data = result.fetchone()
            self.db.commit()

            if chat_data:
                return chat_data.ID_CHAT
            return None

        except Exception as e:
            logger.error(f"Error creating chat with SP: {e}")
            self.db.rollback()
            raise

    def get_chat_by_id(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """
        Get chat by ID

        Args:
            chat_id: Chat identifier

        Returns:
            Dictionary with chat data, or None if not found
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
            """)

            result = self.db.execute(query, {'chat_id': chat_id})
            chat_data = result.fetchone()

            if chat_data:
                return dict(chat_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error getting chat {chat_id}: {e}")
            return None

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

    def update_chat_titulo(
        self,
        chat_id: int,
        titulo: str,
        usumod: str
    ) -> Optional[Dict[str, Any]]:
        """
        Update chat TITULO using stored procedure

        Args:
            chat_id: Chat identifier
            titulo: New chat title
            usumod: Username of the user modifying the chat

        Returns:
            Dictionary with updated chat data, or None if not found
        """
        try:
            # TODO: Replace with actual stored procedure name when available
            # query = text("EXEC dbo.SP_CHAT_UPDATE_TITULO @ID_CHAT = :chat_id, @TITULO = :titulo, @USUMOD = :usumod")

            # Temporary direct UPDATE until SP is created
            query = text("""
                UPDATE dbo.CHATS
                SET
                    TITULO = :titulo,
                    USUMOD = :usumod,
                    FCHMOD = GETDATE()
                OUTPUT INSERTED.ID_CHAT, INSERTED.ID_AREA, INSERTED.ID_EMPRESA, INSERTED.TITULO,
                       INSERTED.ULTIMO_MENSAJE_FECHA, INSERTED.USUCRE, INSERTED.USUMOD,
                       INSERTED.FCHMOD, INSERTED.FCHCRE, INSERTED.ID_ESTADO_REGISTRO
                WHERE ID_CHAT = :chat_id
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'titulo': titulo,
                'usumod': usumod
            })

            chat_data = result.fetchone()
            self.db.commit()

            if chat_data:
                return dict(chat_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error updating chat {chat_id}: {e}")
            self.db.rollback()
            raise

    def delete_chat(self, chat_id: int, usumod: str) -> bool:
        """
        Soft delete chat by setting ID_ESTADO_REGISTRO to 0 using stored procedure

        Args:
            chat_id: Chat identifier
            usumod: Username of the user deleting the chat

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # TODO: Replace with actual stored procedure name when available
            # query = text("EXEC dbo.SP_CHAT_DELETE @ID_CHAT = :chat_id, @USUMOD = :usumod")

            # Temporary direct UPDATE until SP is created
            query = text("""
                UPDATE dbo.CHATS
                SET
                    ID_ESTADO_REGISTRO = 0,
                    USUMOD = :usumod,
                    FCHMOD = GETDATE()
                WHERE ID_CHAT = :chat_id
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'usumod': usumod
            })

            self.db.commit()
            return result.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            self.db.rollback()
            return False

    def get_chat_count(
        self,
        id_empresa: Optional[int] = None,
        id_area: Optional[int] = None,
        id_estado: int = 1
    ) -> int:
        """
        Get total count of chats

        Args:
            id_empresa: Filter by company (optional)
            id_area: Filter by area (optional)
            id_estado: Filter by status (default: 1 = active)

        Returns:
            Total number of chats matching the filters
        """
        try:
            # Build dynamic WHERE clause
            where_conditions = ["ID_ESTADO_REGISTRO = :id_estado"]
            params = {'id_estado': id_estado}

            if id_empresa is not None:
                where_conditions.append("ID_EMPRESA = :id_empresa")
                params['id_empresa'] = id_empresa

            if id_area is not None:
                where_conditions.append("ID_AREA = :id_area")
                params['id_area'] = id_area

            where_clause = " AND ".join(where_conditions)

            query = text(f"""
                SELECT COUNT(*) as total
                FROM dbo.CHATS
                WHERE {where_clause}
            """)

            result = self.db.execute(query, params)
            count_data = result.fetchone()
            return count_data[0] if count_data else 0

        except Exception as e:
            logger.error(f"Error getting chat count: {e}")
            return 0

    def update_ultimo_mensaje_fecha(self, chat_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA to current timestamp

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
            """)

            result = self.db.execute(query, {'chat_id': chat_id})
            self.db.commit()
            return result.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating ultimo_mensaje_fecha for chat {chat_id}: {e}")
            self.db.rollback()
            return False
