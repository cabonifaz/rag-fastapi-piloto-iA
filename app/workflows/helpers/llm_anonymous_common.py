"""
Common LLM helper utilities for anonymous chat workflows.
Handles validation, error handling, and persistence logic for anonymous chats.
"""
import logging
import asyncio
from typing import Any
from app.infrastructure.repositories.chat_anonymous_repository import ChatAnonymousRepository

logger = logging.getLogger(__name__)


async def update_chat_last_message_date_anonymous(db: Any, chat_anonymous_id: str) -> None:
    """
    Update the last message date for an anonymous chat.

    Args:
        db: Database session
        chat_anonymous_id: Anonymous chat identifier
    """
    chat_anonymous_repo = ChatAnonymousRepository(db)
    await asyncio.to_thread(
        chat_anonymous_repo.update_ultimo_mensaje_fecha,
        chat_anonymous_id
    )


async def save_assistant_message_anonymous(
    message_service: Any,
    chat_anonymous_id: str,
    assistant_timestamp: str,
    assistant_response: str,
    store_messages: bool = True
) -> None:
    """
    Save assistant message to DynamoDB for anonymous chat with optional flag.

    Args:
        message_service: Message service for DynamoDB
        chat_anonymous_id: Anonymous chat identifier
        assistant_timestamp: Timestamp for the message
        assistant_response: The assistant's response text
        store_messages: Whether to actually save the message (default: True)
    """
    if store_messages and chat_anonymous_id and assistant_response and assistant_timestamp:
        await message_service.create_message_anonymous(
            chat_anonymous_id=chat_anonymous_id,
            created_at=assistant_timestamp,
            sender=1,
            message=assistant_response
        )
    elif not store_messages:
        logger.info("Skipping anonymous assistant message storage (store_messages=False)")
