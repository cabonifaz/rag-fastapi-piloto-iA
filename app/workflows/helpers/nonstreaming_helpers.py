"""
Non-streaming helper functions for RAG workflows.
Handles complete response generation for N8N integrations.
"""
import logging
from typing import Any
from app.workflows.states import RAGState, LLMOnlyState
from app.workflows.helpers.llm_common import (
    validate_llm_parameters,
    validate_llm_response,
    handle_llm_error,
    update_chat_last_message_date,
    save_assistant_message
)

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

    # Validate parameters using shared utility
    validate_llm_parameters(messages, prompt, max_tokens, temperature)

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

        # Validate response using shared utility
        validate_llm_response(assistant_response)

        # Update chat last message date using shared utility
        await update_chat_last_message_date(db, chat_id)

        # Save assistant message to DynamoDB using shared utility
        await save_assistant_message(
            message_service=message_service,
            chat_id=chat_id,
            assistant_timestamp=state["assistant_timestamp"],
            assistant_response=assistant_response
        )

        return assistant_response

    except Exception as e:
        await handle_llm_error(e, "LLM generation")


async def generate_complete_llm_only_response(
    state: LLMOnlyState,
    llm_only_provider: Any,
    message_service: Any,
    db: Any
) -> str:
    """
    Generate complete LLM-only response (non-streaming), update chat, and save message.
    Consolidates all LLM-only logic including validation and persistence.

    Args:
        state: LLMOnlyState with all necessary context
        llm_only_provider: LLM-only provider
        message_service: Message service for DynamoDB
        db: Database session

    Returns:
        Complete assistant response text

    Raises:
        ValueError: If parameters are invalid or no response generated
        ConnectionError: If LLM service is unavailable
        TimeoutError: If LLM generation times out
    """
    llm_config = state["llm_config"]
    chat_id = state["chat_id"]
    store_messages = state.get("store_messages", True)

    # Extract LLM parameters from state
    model_id = llm_config['config']['LLM_MODEL']
    prompt = state["cleaned_message"]
    max_tokens = llm_config['config']['LLM_MAX_TOKENS']
    temperature = llm_config['config']['LLM_TEMPERATURE']
    top_p = llm_config['config']['LLM_TOP_P']
    role_behavior = state["system_behavior"]
    messages = state.get("conversation_history") or None
    request_timezone = state.get("request_timezone")
    utc_formatted = state.get("utc_formatted")
    local_formatted = state.get("local_formatted")
    use_guidelines = state.get("use_guidelines", True)

    # Validate parameters using shared utility
    validate_llm_parameters(messages, prompt, max_tokens, temperature)

    try:
        # Generate complete response using LLM-only provider
        assistant_response = await llm_only_provider.generate(
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

        # Validate response using shared utility
        validate_llm_response(assistant_response)

        # Update chat last message date using shared utility
        await update_chat_last_message_date(db, chat_id)

        # Save assistant message to DynamoDB using shared utility (respects store_messages flag)
        await save_assistant_message(
            message_service=message_service,
            chat_id=chat_id,
            assistant_timestamp=state["assistant_timestamp"],
            assistant_response=assistant_response,
            store_messages=store_messages
        )

        return assistant_response

    except Exception as e:
        await handle_llm_error(e, "LLM-only generation")
