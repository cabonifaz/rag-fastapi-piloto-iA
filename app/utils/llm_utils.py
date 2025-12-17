from typing import AsyncGenerator, Optional, List, Dict
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate_text_stream_with_validation(
    llm_provider,
    prompt: str = None,
    max_tokens: int = None,
    temperature: float = None,
    role_behavior: str = None,
    messages: Optional[List[Dict[str, str]]] = None,
    timestamp_utc: Optional[str] = None,
    request_timezone: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming text response using LLM provider with validation and user-friendly error handling.

    This utility wraps the LLM provider's generate_stream method to add:
    - Stop reason detection and user-friendly Spanish messages
    - Parameter validation
    - Empty response validation

    Args:
        llm_provider: The LLM provider instance (port implementation)
        prompt: The user prompt (used if messages is None)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        role_behavior: Optional role behavior (system prompt)
        messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                 If provided, prompt will be ignored and messages will be used instead
        timestamp_utc: Optional Unix timestamp in UTC format (as string)
        request_timezone: Optional timezone string for the request

    Yields:
        Text chunks as they are generated

    Raises:
        ValueError: If parameters are invalid or no response generated
        ConnectionError: If LLM service is unavailable
        TimeoutError: If LLM generation times out
    """

    # Validate that either prompt or messages is provided
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")

    # Use provided values or fall back to config defaults
    llm_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    llm_temperature = temperature if temperature is not None else settings.llm_temperature

    # Validate parameters
    if llm_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if not (0.0 <= llm_temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    try:
        has_content = False

        async for chunk in llm_provider.generate_stream(
            prompt=prompt,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
            role_behavior=role_behavior,
            messages=messages,
            timestamp_utc=timestamp_utc,
            request_timezone=request_timezone
        ):
            # Detect stop reason signal from LLM provider
            if chunk.startswith("__STOP_REASON__:"):
                stop_reason = chunk.split(":")[1]
                if stop_reason == "max_tokens":
                    # Yield user-friendly error message in Spanish instead of throwing exception
                    yield "⚠️ El modelo agotó los tokens disponibles durante el análisis de la consulta. Por favor, intenta con una pregunta más específica o reduce la complejidad de tu solicitud."
                    has_content = True
                continue

            has_content = True
            yield chunk

        if not has_content:
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

    except ConnectionError as e:
        logger.error(f"Connection error during LLM generation: {e}")
        raise ConnectionError(f"LLM service unavailable: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input for LLM: {e}")
        raise ValueError(f"Invalid prompt or parameters: {str(e)}")
    except TimeoutError as e:
        logger.error(f"Timeout error during LLM generation: {e}")
        raise TimeoutError(f"LLM generation timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during LLM generation: {e}")
        raise ConnectionError(f"LLM generation failed: {str(e)}")
