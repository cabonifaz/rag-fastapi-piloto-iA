"""Service for managing chat messages in DynamoDB."""

from typing import Optional, List, Dict, Any
import logging

from app.models.message_models import (
    MessageCreate,
    MessageResponse,
    MessageListResponse
)
from app.infrastructure.repositories.message_repository import MessageRepository

logger = logging.getLogger(__name__)


class MessageService:
    """
    Service for message operations in DynamoDB.
    Handles business logic for creating, retrieving, and deleting messages.
    """

    def __init__(self):
        self.repository = MessageRepository()

    async def create_message(
        self,
        request: MessageCreate
    ) -> Optional[MessageResponse]:
        """
        Create a new message in DynamoDB

        Args:
            request: MessageCreate with chat_id, created_at, sender, message

        Returns:
            MessageResponse with created message data
        """
        try:
            message_data = self.repository.create_message(
                chat_id=request.chat_id,
                created_at=request.created_at,
                sender=request.sender,
                message=request.message,
                id_estado_registro=1
            )

            if message_data:
                return self._map_to_message_response(message_data)
            return None

        except Exception as e:
            logger.error(f"Error in create_message service: {e}")
            raise

    async def get_messages_by_chat(
        self,
        chat_id: str,
        limit: int = 50,
        last_evaluated_key: Optional[Dict[str, Any]] = None
    ) -> MessageListResponse:
        """
        Get messages for a specific chat with pagination

        Args:
            chat_id: Chat identifier
            limit: Maximum number of messages to return
            last_evaluated_key: For pagination

        Returns:
            MessageListResponse with messages list and pagination info
        """
        try:
            result = self.repository.get_messages_by_chat(
                chat_id=chat_id,
                limit=limit,
                last_evaluated_key=last_evaluated_key,
                id_estado_registro=1
            )

            messages = [
                self._map_to_message_response(msg)
                for msg in result.get('messages', [])
            ]

            return MessageListResponse(
                messages=messages,
                total_count=result.get('count', 0),
                last_evaluated_key=result.get('last_evaluated_key')
            )

        except Exception as e:
            logger.error(f"Error in get_messages_by_chat service: {e}")
            return MessageListResponse(
                messages=[],
                total_count=0,
                last_evaluated_key=None
            )

    async def get_last_n_messages(
        self,
        chat_id: str,
        n: int = 10
    ) -> List[MessageResponse]:
        """
        Get the last N messages for a chat (most recent)

        Args:
            chat_id: Chat identifier
            n: Number of recent messages to retrieve

        Returns:
            List of MessageResponse (ordered from oldest to newest)
        """
        try:
            messages_data = self.repository.get_last_n_messages(
                chat_id=chat_id,
                n=n,
                id_estado_registro=1
            )

            return [
                self._map_to_message_response(msg)
                for msg in messages_data
            ]

        except Exception as e:
            logger.error(f"Error in get_last_n_messages service: {e}")
            return []

    async def soft_delete_message(
        self,
        chat_id: str,
        created_at: str
    ) -> bool:
        """
        Soft delete a message (set id_estado_registro to 0)

        Args:
            chat_id: Chat identifier
            created_at: Message timestamp

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.repository.soft_delete_message(
                chat_id=chat_id,
                created_at=created_at
            )

        except Exception as e:
            logger.error(f"Error in soft_delete_message service: {e}")
            return False

    async def count_messages(
        self,
        chat_id: str
    ) -> int:
        """
        Count total messages for a chat

        Args:
            chat_id: Chat identifier

        Returns:
            Total count of active messages
        """
        try:
            return self.repository.count_messages(
                chat_id=chat_id,
                id_estado_registro=1
            )

        except Exception as e:
            logger.error(f"Error in count_messages service: {e}")
            return 0

    # =============================================
    # Helper Methods
    # =============================================

    def _map_to_message_response(self, message_data: Dict[str, Any]) -> MessageResponse:
        """Map DynamoDB item to MessageResponse"""
        return MessageResponse(
            chat_id=message_data['chat_id'],
            created_at=message_data['created_at'],
            id_estado_registro=message_data['id_estado_registro'],
            sender=message_data['sender'],
            message=message_data['message'],
            **{'chat_id#id_estado_registro': message_data['chat_id#id_estado_registro']}
        )
