from typing import Optional, List, Dict
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate_text_with_validation(
    llm_provider,
    model_id: str,
    prompt: str = None,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    role_behavior: str = None,
    messages: Optional[List[Dict[str, str]]] = None,
    request_timezone: Optional[str] = None,
    utc_formatted: str = None,
    local_formatted: str = None
) -> str:
    """
    Generate complete text response using non-streaming LLM provider with validation and error handling.

    This utility wraps the LLM provider's generate method to add:
    - Parameter validation
    - Empty response validation
    - Consistent error handling

    Args:
        llm_provider: The LLM provider instance (non-streaming port implementation)
        model_id: Model ID to use (from database config)
        prompt: The user prompt (used if messages is None)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p (nucleus) sampling parameter
        role_behavior: Optional role behavior (system prompt)
        messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                 If provided, prompt will be ignored and messages will be used instead
        request_timezone: Optional timezone string for the request
        utc_formatted: Formatted UTC timestamp string
        local_formatted: Formatted local timestamp string

    Returns:
        Complete generated text as a single string

    Raises:
        ValueError: If parameters are invalid or no response generated
        ConnectionError: If LLM service is unavailable
        TimeoutError: If LLM generation times out
    """

    # Validate that either prompt or messages is provided
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")

    # Validate parameters
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    try:
        # Call non-streaming provider
        response = await llm_provider.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            role_behavior=role_behavior,
            messages=messages,
            request_timezone=request_timezone,
            utc_formatted=utc_formatted,
            local_formatted=local_formatted
        )

        # Validate response
        if not response or not response.strip():
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        return response

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


async def generate_text_llm_only_with_validation(
    llm_provider,
    model_id: str,
    prompt: str = None,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    role_behavior: str = None,
    messages: Optional[List[Dict[str, str]]] = None,
    request_timezone: Optional[str] = None,
    utc_formatted: str = None,
    local_formatted: str = None,
    use_guidelines: bool = True
) -> str:
    """
    Generate complete text response using non-streaming LLM-only provider with validation and error handling.

    This utility is specifically for LLM-only mode (no RAG) and supports the use_guidelines parameter.

    Args:
        llm_provider: The LLM-only provider instance (non-streaming port implementation)
        model_id: Model ID to use (from database config)
        prompt: The user prompt (used if messages is None)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p (nucleus) sampling parameter
        role_behavior: Optional role behavior (system prompt)
        messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                 If provided, prompt will be ignored and messages will be used instead
        request_timezone: Optional timezone string for the request
        utc_formatted: Formatted UTC timestamp string
        local_formatted: Formatted local timestamp string
        use_guidelines: Whether to include conversational guidelines in system prompt (default: True)

    Returns:
        Complete generated text as a single string

    Raises:
        ValueError: If parameters are invalid or no response generated
        ConnectionError: If LLM service is unavailable
        TimeoutError: If LLM generation times out
    """

    # Validate that either prompt or messages is provided
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")

    # Validate parameters
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    try:
        # Call non-streaming LLM-only provider
        response = await llm_provider.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            role_behavior=role_behavior,
            messages=messages,
            request_timezone=request_timezone,
            utc_formatted=utc_formatted,
            local_formatted=local_formatted,
            use_guidelines=use_guidelines
        )

        # Validate response
        if not response or not response.strip():
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        return response

    except ConnectionError as e:
        logger.error(f"Connection error during LLM-only generation: {e}")
        raise ConnectionError(f"LLM service unavailable: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input for LLM-only: {e}")
        raise ValueError(f"Invalid prompt or parameters: {str(e)}")
    except TimeoutError as e:
        logger.error(f"Timeout error during LLM-only generation: {e}")
        raise TimeoutError(f"LLM generation timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during LLM-only generation: {e}")
        raise ConnectionError(f"LLM generation failed: {str(e)}")
