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

    async def get_messages_by_chat(
        self,
        chat_id: str,
        limit: int = 20,
        last_evaluated_key: Optional[Dict[str, Any]] = None
    ) -> MessageListResponse:
        """
        Get messages for a specific chat with pagination.

        Args:
            chat_id: Chat identifier
            limit: Maximum number of messages to return
            last_evaluated_key: For pagination.

        Returns:
            MessageListResponse with a list of messages and pagination info.
        """
        try:
            # Delegate the database call to the repository
            response_data = self.repository.get_messages_by_chat(
                chat_id=chat_id,
                limit=limit,
                last_evaluated_key=last_evaluated_key
            )

            # Map the raw data to Pydantic models
            messages = [
                self._map_to_message_response(msg)
                for msg in response_data.get('messages', [])
            ]

            # Get the total count of messages for the entire chat for accurate pagination
            total_count = self.repository.count_messages(chat_id=chat_id, id_estado_registro=1)

            return MessageListResponse(
                messages=messages,
                total_count=total_count,
                last_evaluated_key=response_data.get('last_evaluated_key')
            )

        except Exception as e:
            logger.error(f"Error getting messages for chat_id {chat_id}: {e}")
            # Return a properly structured empty response on error
            return MessageListResponse(messages=[], total_count=0, last_evaluated_key=None)

    async def create_message(
        self,
        chat_id: int,
        created_at: str,
        sender: int,
        message: str,
        id_estado_registro: int = 1
    ) -> bool:
        """
        Create a new message in DynamoDB.

        Args:
            chat_id: Chat identifier (integer)
            created_at: Timestamp as string (milliseconds since epoch)
            sender: 0 = user, 1 = assistant
            message: Message content
            id_estado_registro: Status (default: 1 = active)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.repository.create_message(
                chat_id=chat_id,
                created_at=created_at,
                sender=sender,
                message=message,
                id_estado_registro=id_estado_registro
            )
            logger.info(f"Message created successfully for chat_id: {chat_id}")
            return True

        except Exception as e:
            logger.error(f"Error in create_message service: {e}")
            return False

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
            # Use get_last_n_messages_by_chat instead of get_last_n_messages
            response_data = self.repository.get_last_n_messages_by_chat(
                chat_id=chat_id,
                limit=n
            )

            return [
                self._map_to_message_response(msg)
                for msg in response_data.get('messages', [])
            ]

        except Exception as e:
            logger.error(f"Error in get_last_n_messages service: {e}")
            return []


    # =============================================
    # Helper Methods
    # =============================================

    def _map_to_message_response(self, message_data: Dict[str, Any]) -> MessageResponse:
        """Map DynamoDB item to the simplified MessageResponse model."""
        return MessageResponse(
            id=message_data['created_at'],  # Use created_at as the unique ID for the frontend
            chat_id=message_data['chat_id'],
            created_at=message_data['created_at'],
            sender=int(message_data['sender']),  # Ensure sender is an int
            message=message_data['message']
        )
