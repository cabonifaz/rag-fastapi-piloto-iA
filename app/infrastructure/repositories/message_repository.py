"""Repository for DynamoDB message operations."""

import boto3
from boto3.dynamodb.conditions import Key
from typing import Optional, List, Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class MessageRepository:
    """
    Repository for DynamoDB MESSAGES table operations.

    Table structure:
    - chat_id (Partition Key): STRING
    - created_at (Sort Key): STRING (timestamp in milliseconds)
    - id_estado_registro: NUMBER (1 = active, 0 = deleted)
    - chat_id#id_estado_registro (GSI): STRING (composite key for filtering by status)
    - sender: NUMBER (0 = user, 1 = assistant)
    - message: STRING
    """

    def __init__(self):
        """Initialize DynamoDB client using AWS profile from settings"""
        session = boto3.Session(
            profile_name=settings.aws_profile,
            region_name=settings.aws_region
        )
        self.dynamodb = session.resource('dynamodb')
        self.table = self.dynamodb.Table(settings.dynamodb_table_messages)
        self.table_name = settings.dynamodb_table_messages

    def create_message(
        self,
        chat_id: str,
        created_at: str,
        sender: int,
        message: str,
        id_estado_registro: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new message in DynamoDB

        Args:
            chat_id: Chat identifier
            created_at: Timestamp as string (milliseconds since epoch)
            sender: 0 = user, 1 = assistant
            message: Message content
            id_estado_registro: Status (default: 1 = active)

        Returns:
            Dictionary with created message data
        """
        try:
            # Create composite key for GSI
            chat_id_estado = f"{chat_id}#{id_estado_registro}"

            item = {
                'chat_id': chat_id,
                'created_at': created_at,
                'id_estado_registro': id_estado_registro,
                'chat_id#id_estado_registro': chat_id_estado,
                'sender': sender,
                'message': message
            }

            self.table.put_item(Item=item)
            logger.info(f"Message created successfully for chat_id: {chat_id}")
            return item

        except Exception as e:
            logger.error(f"Error creating message for chat_id {chat_id}: {e}")
            raise

    def get_messages_by_chat(
        self,
        chat_id: str,
        limit: int = 50,
        last_evaluated_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get messages for a specific chat with pagination

        Args:
            chat_id: Chat identifier (e.g., "chat-123")
            limit: Maximum number of messages to return
            last_evaluated_key: For pagination (from previous response)

        Returns:
            Dictionary with messages list and pagination info
        """
        try:
            # Ensure chat_id has the "chat-" prefix
            if not chat_id.startswith('chat-'):
                chat_id = f"chat-{chat_id}"

            gsi_pk_value = f"{chat_id}#1"

            query_params = {
                'IndexName': 'chat_id_id_estado_registro_created_at_index',
                'KeyConditionExpression': Key('chat_id#id_estado_registro').eq(gsi_pk_value),
                'Limit': limit,
                'ScanIndexForward': True  # True = ascending order (oldest first)
            }

            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key

            response = self.table.query(**query_params)

            return {
                # Return messages as is (newest to oldest)
                'messages': response.get('Items', []),
                'count': response.get('Count', 0),
                'last_evaluated_key': response.get('LastEvaluatedKey')
            }

        except Exception as e:
            logger.error(f"Error getting messages for chat_id {chat_id}: {e}")
            return {'messages': [], 'count': 0, 'last_evaluated_key': None}

    def get_message(
        self,
        chat_id: str,
        created_at: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific message by chat_id and created_at

        Args:
            chat_id: Chat identifier
            created_at: Message timestamp

        Returns:
            Message data or None if not found
        """
        try:
            response = self.table.get_item(
                Key={
                    'chat_id': chat_id,
                    'created_at': created_at
                }
            )

            return response.get('Item')

        except Exception as e:
            logger.error(f"Error getting message {chat_id}#{created_at}: {e}")
            return None

    def soft_delete_message(
        self,
        chat_id: str,
        created_at: str
    ) -> bool:
        """
        Soft delete a message by setting id_estado_registro to 0

        Args:
            chat_id: Chat identifier
            created_at: Message timestamp

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update composite key
            chat_id_estado = f"{chat_id}#0"

            self.table.update_item(
                Key={
                    'chat_id': chat_id,
                    'created_at': created_at
                },
                UpdateExpression='SET id_estado_registro = :estado, #gsi_key = :gsi_value',
                ExpressionAttributeNames={
                    '#gsi_key': 'chat_id#id_estado_registro'
                },
                ExpressionAttributeValues={
                    ':estado': 0,
                    ':gsi_value': chat_id_estado
                }
            )

            logger.info(f"Message soft deleted: {chat_id}#{created_at}")
            return True

        except Exception as e:
            logger.error(f"Error soft deleting message {chat_id}#{created_at}: {e}")
            return False

    def delete_message(
        self,
        chat_id: str,
        created_at: str
    ) -> bool:
        """
        Hard delete a message (permanent deletion)

        Args:
            chat_id: Chat identifier
            created_at: Message timestamp

        Returns:
            True if successful, False otherwise
        """
        try:
            self.table.delete_item(
                Key={
                    'chat_id': chat_id,
                    'created_at': created_at
                }
            )

            logger.info(f"Message permanently deleted: {chat_id}#{created_at}")
            return True

        except Exception as e:
            logger.error(f"Error deleting message {chat_id}#{created_at}: {e}")
            return False

    def get_last_n_messages(
        self,
        chat_id: str,
        n: int = 10,
        id_estado_registro: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get the last N messages for a chat (most recent)

        Args:
            chat_id: Chat identifier
            n: Number of recent messages to retrieve
            id_estado_registro: Filter by status (default: 1 = active)

        Returns:
            List of messages (ordered from oldest to newest)
        """
        try:
            response = self.table.query(
                KeyConditionExpression=Key('chat_id').eq(chat_id),
                FilterExpression=boto3.dynamodb.conditions.Attr('id_estado_registro').eq(id_estado_registro),
                Limit=n,
                ScanIndexForward=False  # False = descending order (newest first)
            )

            messages = response.get('Items', [])

            # Reverse to return oldest to newest
            return list(reversed(messages))

        except Exception as e:
            logger.error(f"Error getting last {n} messages for chat_id {chat_id}: {e}")
            return []

    def count_messages(
        self,
        chat_id: str,
        id_estado_registro: int = 1
    ) -> int:
        """
        Count total messages for a chat

        Args:
            chat_id: Chat identifier
            id_estado_registro: Filter by status (default: 1 = active)

        Returns:
            Total count of messages
        """
        try:
            response = self.table.query(
                KeyConditionExpression=Key('chat_id').eq(chat_id),
                FilterExpression=boto3.dynamodb.conditions.Attr('id_estado_registro').eq(id_estado_registro),
                Select='COUNT'
            )

            return response.get('Count', 0)

        except Exception as e:
            logger.error(f"Error counting messages for chat_id {chat_id}: {e}")
            return 0
