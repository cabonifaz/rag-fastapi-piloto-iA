"""Service for managing chat messages in DynamoDB."""

from typing import Optional, List, Dict, Any
import logging

from app.models.message_models import (
    MessageCreate,
    MessageResponse,
    MessageListResponse
)
from app.infrastructure.repositories.message_repository import MessageRepository
from app.domain.ports.blob_storage_port import BlobStoragePort
from app.core.config import settings

logger = logging.getLogger(__name__)


class MessageService:
    """
    Service for message operations in DynamoDB.
    Handles business logic for creating, retrieving, and deleting messages.
    """

    def __init__(self):
        self.repository = MessageRepository()
        # ParametrosService is used to fetch dynamic limits from DB (e.g., SP_PARAMETROS_LST '11')
        from app.services.parametros_service import ParametrosService
        self.parametros_service = ParametrosService()

    async def get_messages_by_chat(
        self,
        chat_id: str,
        last_evaluated_key: Optional[Dict[str, Any]] = None,
        db: object = None
    ) -> MessageListResponse:
        """
        Get messages for a specific chat with pagination.

        Args:
            chat_id: Chat identifier
            last_evaluated_key: For pagination.
            db: Optional SQLAlchemy Session used to read parameters

        Returns:
            MessageListResponse with a list of messages and pagination info.
        """
        try:
            try:
                params = await self.parametros_service.get_param_by_id_maestro(db, "11")
                row = next((p for p in params if p.get('ID_MAESTRO') == 11), None)
                limit = row['NUM1'] if row else 15
            except Exception:
                limit = 15

            response_data = await self.repository.get_messages_by_chat(
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
            total_count = await self.repository.count_messages(chat_id=chat_id, id_estado_registro=1)

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
            await self.repository.create_message(
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

    async def create_message_with_attachments(
        self,
        chat_id: int,
        created_at: str,
        sender: int,
        message: str,
        attachment_keys: List[str],
        id_estado_registro: int = 1
    ) -> bool:
        """
        Create a new message with attachment S3 keys in DynamoDB.

        Args:
            chat_id: Chat identifier (integer)
            created_at: Timestamp as string (milliseconds since epoch)
            sender: 0 = user, 1 = assistant
            message: Message content
            attachment_keys: List of S3 object keys for attached files
            id_estado_registro: Status (default: 1 = active)

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.repository.create_message_with_attachments(
                chat_id=chat_id,
                created_at=created_at,
                sender=sender,
                message=message,
                attachment_keys=attachment_keys,
                id_estado_registro=id_estado_registro
            )
            logger.info(f"Message with attachments created successfully for chat_id: {chat_id}")
            return True

        except Exception as e:
            logger.error(f"Error in create_message_with_attachments service: {e}")
            return False

    async def get_last_n_messages(
        self,
        chat_id: str,
        n: int
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
            response_data = await self.repository.get_last_n_messages_by_chat(
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

    async def create_message_anonymous(
        self,
        chat_anonymous_id: int,
        created_at: str,
        sender: int,
        message: str,
        id_estado_registro: int = 1
    ) -> bool:
        """
        Create a new message in DynamoDB for anonymous chat.

        Args:
            chat_anonymous_id: Anonymous chat identifier (integer)
            created_at: Timestamp as string (milliseconds since epoch)
            sender: 0 = user, 1 = assistant
            message: Message content
            id_estado_registro: Status (default: 1 = active)

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.repository.create_message_anonymous(
                chat_anonymous_id=chat_anonymous_id,
                created_at=created_at,
                sender=sender,
                message=message,
                id_estado_registro=id_estado_registro
            )
            logger.info(f"Anonymous message created successfully for chat_anonymous_id: {chat_anonymous_id}")
            return True

        except Exception as e:
            logger.error(f"Error in create_message_anonymous service: {e}")
            return False

    async def get_last_n_messages_anonymous(
        self,
        chat_anonymous_id: str,
        n: int
    ) -> List[MessageResponse]:
        """
        Get the last N messages for an anonymous chat (most recent)

        Args:
            chat_anonymous_id: Anonymous chat identifier
            n: Number of recent messages to retrieve

        Returns:
            List of MessageResponse (ordered from oldest to newest)
        """
        try:
            response_data = await self.repository.get_last_n_messages_by_chat_anonymous(
                chat_anonymous_id=chat_anonymous_id,
                limit=n
            )

            return [
                self._map_to_message_response(msg)
                for msg in response_data.get('messages', [])
            ]

        except Exception as e:
            logger.error(f"Error in get_last_n_messages_anonymous service: {e}")
            return []


    async def get_attachment_urls(
        self,
        blob_storage: BlobStoragePort,
        attachment_keys: List[str],
        download: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned GET URLs for chat attachment keys stored in a message.

        Args:
            blob_storage: Blob storage port instance
            attachment_keys: List of S3 keys (e.g. ["chats/2/1234/file.jpeg"])
            download: If True, forces file download instead of inline view

        Returns:
            List of objects with presigned_url, s3_key, and filename
        """
        bucket = settings.s3_chat_files

        presigned_urls = await blob_storage.generate_presigned_download_urls_batch(
            bucket_name=bucket,
            object_keys=attachment_keys,
            expiration_seconds=300,
            as_attachment=download,
        )

        return [
            {"presigned_url": url, "s3_key": key, "filename": key.split("/")[-1]}
            for url, key in zip(presigned_urls, attachment_keys)
        ]

    async def generate_attachment_presigned_urls(
        self,
        blob_storage: BlobStoragePort,
        user_id: str,
        timestamp: str,
        filenames: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned PUT URLs for chat attachment uploads.

        S3 path: chats/{user_id}/{timestamp}/{filename}
        Bucket: S3_CHAT_FILES

        Args:
            blob_storage: Blob storage port instance
            user_id: User identifier
            timestamp: Timestamp string (used as folder prefix)
            filenames: List of filenames to upload

        Returns:
            List of objects with presigned_url, s3_key, and filename
        """
        bucket = settings.s3_chat_files
        s3_keys = [f"chats/{user_id}/{timestamp}/{filename}" for filename in filenames]

        presigned_urls = await blob_storage.generate_presigned_upload_urls_batch(
            bucket_name=bucket,
            object_keys=s3_keys,
            expiration_seconds=300,
        )

        return [
            {"presigned_url": url, "s3_key": key, "filename": filename}
            for url, key, filename in zip(presigned_urls, s3_keys, filenames)
        ]

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
            message=message_data['message'],
            attachment_keys=message_data.get('attachment_keys')
        )
