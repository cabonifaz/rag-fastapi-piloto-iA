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
        chat_id: int,
        created_at: str,
        sender: int,
        message: str,
        id_estado_registro: int = 1
    ) -> None:
        """
        Save a message to DynamoDB.

        Args:
            chat_id: Chat identifier
            created_at: Timestamp as string (milliseconds since epoch)
            sender: 0 = user, 1 = assistant
            message: Message content
            id_estado_registro: Status (default: 1 = active)

        Raises:
            Exception: If message save fails
        """
        try:
            # Initialize DynamoDB client
            session = boto3.Session(
                profile_name=settings.aws_profile,
                region_name=settings.aws_region
            )
            dynamodb = session.resource('dynamodb')
            table = dynamodb.Table(settings.dynamodb_table_messages)

            # Convert chat_id to DynamoDB format: "chat-{id}"
            chat_id_str = f"chat-{chat_id}"

            # Create composite key for GSI
            chat_id_estado = f"{chat_id_str}#{id_estado_registro}"

            item = {
                'chat_id': chat_id_str,
                'created_at': created_at,
                'id_estado_registro': id_estado_registro,
                'chat_id#id_estado_registro': chat_id_estado,
                'sender': sender,
                'message': message
            }

            table.put_item(Item=item)

        except Exception as e:
            logger.error(f"Error saving message for chat_id {chat_id}: {e}")
            raise

    def get_messages_by_chat(
        self,
        chat_id: str,
        limit: int = 20,
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

    def get_last_n_messages_by_chat(
        self,
        chat_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get messages for a specific chat without pagination

        Args:
            chat_id: Chat identifier (e.g., "chat-123")
            limit: Maximum number of messages to return

        Returns:
            Dictionary with messages list and count
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
                'ScanIndexForward': False
            }

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