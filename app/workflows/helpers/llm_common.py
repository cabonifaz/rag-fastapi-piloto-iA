"""
Common LLM helper utilities shared across streaming and non-streaming workflows.
Consolidates validation, error handling, and persistence logic.
"""
import logging
import asyncio
from typing import Any, Optional, List, Dict
from app.infrastructure.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)


def validate_llm_parameters(
    messages: Optional[List[Dict[str, str]]],
    prompt: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float]
) -> None:
    """
    Validate LLM generation parameters.

    Args:
        messages: Optional conversation history
        prompt: Optional prompt string
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate that either prompt or messages is provided
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")

    # Validate max_tokens
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    # Validate temperature
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")


async def handle_llm_error(error: Exception, context: str = "LLM generation") -> None:
    """
    Handle and re-raise LLM errors with appropriate logging and error types.

    Args:
        error: The exception to handle
        context: Context string for logging (e.g., "LLM generation", "LLM-only generation")

    Raises:
        ConnectionError: If LLM service is unavailable
        ValueError: If input parameters are invalid
        TimeoutError: If LLM generation times out
    """
    if isinstance(error, ConnectionError):
        logger.error(f"Connection error during {context}: {error}")
        raise ConnectionError(f"LLM service unavailable: {str(error)}")
    elif isinstance(error, ValueError):
        logger.error(f"Invalid input for {context}: {error}")
        raise ValueError(f"Invalid prompt or parameters: {str(error)}")
    elif isinstance(error, TimeoutError):
        logger.error(f"Timeout error during {context}: {error}")
        raise TimeoutError(f"LLM generation timeout: {str(error)}")
    else:
        logger.error(f"Unexpected error during {context}: {error}")
        raise ConnectionError(f"LLM generation failed: {str(error)}")


async def update_chat_last_message_date(db: Any, chat_id: str) -> None:
    """
    Update the last message date for a chat.

    Args:
        db: Database session
        chat_id: Chat identifier
    """
    chat_repo = ChatRepository(db)
    await asyncio.to_thread(
        chat_repo.update_ultimo_mensaje_fecha,
        chat_id
    )


async def save_assistant_message(
    message_service: Any,
    chat_id: str,
    assistant_timestamp: str,
    assistant_response: str,
    store_messages: bool = True
) -> None:
    """
    Save assistant message to DynamoDB with optional flag.

    Args:
        message_service: Message service for DynamoDB
        chat_id: Chat identifier
        assistant_timestamp: Timestamp for the message
        assistant_response: The assistant's response text
        store_messages: Whether to actually save the message (default: True)
    """
    if store_messages and chat_id and assistant_response and assistant_timestamp:
        await message_service.create_message(
            chat_id=chat_id,
            created_at=assistant_timestamp,
            sender=1,
            message=assistant_response
        )
    elif not store_messages:
        logger.info("Skipping assistant message storage (store_messages=False)")


def validate_llm_response(response: str) -> None:
    """
    Validate that LLM generated a non-empty response.

    Args:
        response: The LLM response to validate

    Raises:
        ValueError: If response is empty or whitespace-only
    """
    if not response or not response.strip():
        raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")
