"""
Streaming helper functions for RAG workflows.
Handles progress streaming and LLM response streaming.
"""
import logging
from typing import Any, AsyncGenerator, Tuple
from app.workflows.states import RAGState
from app.workflows.helpers.llm_common import (
    validate_llm_parameters,
    handle_llm_error,
    update_chat_last_message_date,
    save_assistant_message
)

logger = logging.getLogger(__name__)


async def stream_workflow_progress(app: Any, initial_state: RAGState) -> AsyncGenerator[Tuple[str, Any], None]:
    """
    Execute workflow and yield progress updates.
    Generator that yields progress events during execution, then yields final state.

    Yields:
        Tuples of (event_type, event_data):
        - ("progress", {"type": "progress", "message": "..."})
        - ("state", final_state)
    """
    result_state = None
    progress_sent = {
        "initial": False,
        "analyzing": False,
        "searching": False,
        "preparing": False
    }

    async for state in app.astream(initial_state, stream_mode="values"):
        result_state = state

        # Send initial progress on very first state update
        if not progress_sent["initial"]:
            yield ("progress", {
                "type": "progress",
                "message": "Procesando consulta..."
            })
            progress_sent["initial"] = True

        # Stage 1: Query processing
        if state.get("cleaned_message") and not progress_sent["analyzing"]:
            yield ("progress", {
                "type": "progress",
                "message": "Analizando tu pregunta..."
            })
            progress_sent["analyzing"] = True

        # Stage 2: Searching documents
        elif state.get("query_embedding") and not progress_sent["searching"]:
            yield ("progress", {
                "type": "progress",
                "message": "Buscando información relevante..."
            })
            progress_sent["searching"] = True

        # Stage 3: Preparing response
        elif state.get("rag_prompt") and not progress_sent["preparing"]:
            yield ("progress", {
                "type": "progress",
                "message": "Preparando respuesta..."
            })
            progress_sent["preparing"] = True

    # Yield final state
    yield ("state", result_state)


async def stream_llm_response(
    state: RAGState,
    llm_provider: Any,
    message_service: Any,
    db: Any
) -> AsyncGenerator[dict, None]:
    """
    Stream LLM response and handle message saving.
    Consolidates all streaming logic including validation, error handling, and persistence.

    Yields:
        Dictionary events with type "chunk" and content
    """
    rag_config = state["rag_config"]
    chat_id = state["chat_id"]
    assistant_response = ""
    first_chunk_sent = False

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
        has_content = False

        # Stream LLM response with validation
        async for chunk in llm_provider.generate_stream(
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
        ):
            # Detect stop reason signal from LLM provider
            if chunk.startswith("__STOP_REASON__:"):
                stop_reason = chunk.split(":")[1]
                if stop_reason == "max_tokens":
                    # Yield user-friendly error message in Spanish
                    error_msg = "⚠️ El modelo agotó los tokens disponibles durante el análisis de la consulta. Por favor, intenta con una pregunta más específica o reduce la complejidad de tu solicitud."
                    assistant_response += error_msg
                    has_content = True
                    yield {
                        "type": "chunk",
                        "content": error_msg
                    }
                continue

            has_content = True
            assistant_response += chunk

            # Update chat last message date on first chunk
            if not first_chunk_sent and chunk.strip():
                await update_chat_last_message_date(db, chat_id)
                first_chunk_sent = True

            yield {
                "type": "chunk",
                "content": chunk
            }

        # Validate that content was generated
        if not has_content:
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        # Save assistant message to DynamoDB
        await save_assistant_message(
            message_service=message_service,
            chat_id=chat_id,
            assistant_timestamp=state["assistant_timestamp"],
            assistant_response=assistant_response
        )

    except Exception as e:
        await handle_llm_error(e, "LLM generation")
