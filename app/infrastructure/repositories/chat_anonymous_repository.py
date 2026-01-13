"""Repository for anonymous chat database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class ChatAnonymousRepository:
    """
    Repository for CHATS_ANONIMOS table operations.
    """

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def update_ultimo_mensaje_fecha(self, chat_anonymous_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA using stored procedure SP_UPDATE_CHAT_ANONIMO_ULTIMO_MENSAJE_FECHA

        Args:
            chat_anonymous_id: Anonymous chat identifier

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            query = text("""
                EXEC SP_UPDATE_CHAT_ANONIMO_ULTIMO_MENSAJE_FECHA
                @ID_CHAT_ANONIMO = :id_chat_anonimo
            """)

            self.db.execute(query, {'id_chat_anonimo': chat_anonymous_id})
            self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating ultimo_mensaje_fecha for anonymous chat {chat_anonymous_id}: {e}")
            self.db.rollback()
            return False
