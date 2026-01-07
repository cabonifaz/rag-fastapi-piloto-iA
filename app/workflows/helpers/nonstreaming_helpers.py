"""
Non-streaming helper functions for RAG workflows.
Handles complete response generation for N8N integrations.
"""
import logging
import asyncio
from typing import Any
from app.workflows.states import RAGState
from app.infrastructure.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)


async def execute_workflow_n8n(app: Any, initial_state: RAGState) -> RAGState:
    """
    Execute workflow without streaming progress updates.
    Returns final state directly.

    Args:
        app: Compiled workflow application
        initial_state: Initial RAGState

    Returns:
        Final RAGState after workflow execution
    """
    result_state = await app.ainvoke(initial_state)
    return result_state


async def generate_complete_llm_response(
    state: RAGState,
    llm_nonstreaming_provider: Any,
    message_service: Any,
    db: Any
) -> str:
    """
    Generate complete LLM response (non-streaming), update chat, and save message.
    Consolidates all non-streaming LLM logic including validation and persistence.

    Args:
        state: RAGState with all necessary context
        llm_nonstreaming_provider: Non-streaming LLM provider
        message_service: Message service for DynamoDB
        db: Database session

    Returns:
        Complete assistant response text

    Raises:
        ValueError: If parameters are invalid or no response generated
        ConnectionError: If LLM service is unavailable
        TimeoutError: If LLM generation times out
    """
    rag_config = state["rag_config"]
    chat_id = state["chat_id"]

    # Extract LLM parameters from state
    model_id = rag_config['config']['LLM_MODEL']
    prompt = state["rag_prompt"]
    max_tokens = rag_config['config']['LLM_MAX_TOKENS']
    temperature = rag_config['config']['LLM_TEMPERATURE']
    top_p = rag_config['config']['LLM_TOP_P']
    role_behavior = rag_config['config']['ROLE_BEHAVIOR']
    messages = state.get("conversation_history_for_prompt") or None
    request_timezone = state.get("request_timezone")
    utc_formatted = state.get("utc_formatted")
    local_formatted = state.get("local_formatted")

    # Validate parameters
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    try:
        # Generate complete response using non-streaming provider
        assistant_response = await llm_nonstreaming_provider.generate(
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
        if not assistant_response or not assistant_response.strip():
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        # Update chat last message date
        chat_repo = ChatRepository(db)
        await asyncio.to_thread(
            chat_repo.update_ultimo_mensaje_fecha,
            chat_id
        )

        # Save assistant message to DynamoDB
        if chat_id and assistant_response and state["assistant_timestamp"]:
            await message_service.create_message(
                chat_id=chat_id,
                created_at=state["assistant_timestamp"],
                sender=1,
                message=assistant_response
            )

        return assistant_response

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
