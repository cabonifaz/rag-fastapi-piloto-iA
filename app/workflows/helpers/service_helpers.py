"""
Service layer helper functions.
Utilities for building state, validation, and event generation.
"""
from typing import Optional
from app.workflows.states import RAGState


def build_initial_state(
    user_id: int,
    message: str,
    company_id: int,
    area_id: int,
    created_at: str,
    chat_id: Optional[str] = None,
    request_timezone: Optional[str] = None
) -> RAGState:
    """Build initial state for workflow execution"""
    return {
        # Input parameters
        "user_id": user_id,
        "message": message,
        "company_id": company_id,
        "area_id": area_id,
        "created_at": created_at,
        "chat_id": chat_id,
        "request_timezone": request_timezone,
        # Processing state (will be populated by workflow)
        "cleaned_message": None,
        "conversation_history": [],
        "state_builder_result": None,
        "query_rewriter_result": None,
        "rag_config": None,
        "new_chat_created": False,
        "new_chat_titulo": None,
        "new_chat_timestamp": None,
        "query_for_search": None,
        "query_embedding": None,
        "search_results": None,
        "context_text": None,
        "conversation_history_for_prompt": [],
        "rag_prompt": None,
        "assistant_timestamp": None,
        "assistant_timestamp_ms": None,
        "utc_formatted": None,
        "local_formatted": None,
        # Error handling
        "error": None,
        "should_stop": False
    }


def validate_workflow_state(state: Optional[RAGState]) -> RAGState:
    """
    Validate that workflow completed successfully.
    Raises ValueError if state is invalid.
    """
    if state is None or state.get("should_stop", False):
        error_msg = state.get("error", "Unknown error") if state else "Workflow did not complete"
        raise ValueError(error_msg)

    # Validate required fields
    if not state.get("chat_id") or not state.get("rag_config") or not state.get("rag_prompt"):
        missing_fields = []
        if not state.get("chat_id"):
            missing_fields.append("chat_id")
        if not state.get("rag_config"):
            missing_fields.append("rag_config")
        if not state.get("rag_prompt"):
            missing_fields.append("rag_prompt")
        raise ValueError(f"Workflow incomplete: missing {', '.join(missing_fields)}")

    return state


def generate_metadata_events(state: RAGState, area_id: int, company_id: int):
    """Generate metadata events from workflow state"""
    events = []

    # Metadata
    events.append({
        "type": "metadata",
        "chat_id": state["chat_id"]
    })

    # Chat created event
    if state.get("new_chat_created", False):
        events.append({
            "type": "chat_created",
            "chat": {
                "ID_CHAT": state["chat_id"],
                "ID_AREA": area_id,
                "ID_EMPRESA": company_id,
                "TITULO": state["new_chat_titulo"],
                "ULTIMO_MENSAJE_FECHA": state["new_chat_timestamp"],
                "ID_ESTADO_REGISTRO": 1
            }
        })

    # Assistant metadata
    events.append({
        "type": "assistant_metadata",
        "sender": 1,
        "created_at": state["assistant_timestamp"]
    })

    return events
