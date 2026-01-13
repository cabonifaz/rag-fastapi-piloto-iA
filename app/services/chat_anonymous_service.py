"""Service for managing anonymous chat sessions."""

from sqlalchemy.orm import Session
import logging

from app.infrastructure.repositories.chat_anonymous_repository import ChatAnonymousRepository

logger = logging.getLogger(__name__)


class ChatAnonymousService:
    """
    Service for anonymous chat operations.
    Handles business logic for anonymous chats.
    """

    async def update_ultimo_mensaje_fecha(self, db: Session, chat_anonymous_id: int) -> bool:
        """
        Update ULTIMO_MENSAJE_FECHA to current timestamp for anonymous chat
        Should be called when a new message is added to the anonymous chat

        Args:
            db: Database session
            chat_anonymous_id: Anonymous chat identifier

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            repository = ChatAnonymousRepository(db)
            return repository.update_ultimo_mensaje_fecha(chat_anonymous_id)
        except Exception as e:
            logger.error(f"Error in update_ultimo_mensaje_fecha service for anonymous chat: {e}")
            return False
