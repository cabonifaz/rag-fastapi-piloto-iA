"""
LangGraph workflow for LLM-only query processing for anonymous chats (no RAG).
Reuses existing nodes with different parameters for minimal code duplication.
"""
from typing import Any, Optional
from langgraph.graph import StateGraph, END
import logging

from app.workflows.states import LLMOnlyAnonymousState
from app.workflows.nodes.validation_nodes import create_validate_inputs_node_anonymous
from app.workflows.nodes.conversation_anonymous_nodes import create_get_conversation_history_anonymous_node
from app.workflows.nodes.preprocessing_nodes import create_clean_message_node, create_load_rag_config_node
from app.workflows.nodes.chat_anonymous_nodes import create_save_user_message_anonymous_node
from app.workflows.nodes.prompt_nodes import create_prepare_timestamps_node

logger = logging.getLogger(__name__)

# Global compiled workflow - initialized at startup
_compiled_llm_only_anonymous_workflow: Optional[Any] = None


def create_llm_only_anonymous_workflow(
    session_factory: Any,
    message_service: Any,
    ia_config_service: Any
) -> StateGraph:
    """
    Create and configure the LLM-only anonymous workflow graph (no RAG).
    Reuses existing nodes with different parameters.

    Args:
        session_factory: Database session factory (SessionLocal)
        message_service: Message service
        ia_config_service: IA config service

    Returns:
        Compiled StateGraph ready to execute
    """
    # Create nodes using existing factory functions with LLM-only parameters
    validate_inputs = create_validate_inputs_node_anonymous(require_area_id=False)  # No area required
    get_conversation_history_anonymous = create_get_conversation_history_anonymous_node(message_service, max_ctx=8)  # Max 8 for LLM-only
    clean_message = create_clean_message_node()  # Reuse as-is
    load_config = create_load_rag_config_node(session_factory, ia_config_service, area_required=False)  # Load LLM config
    save_user_message_anonymous = create_save_user_message_anonymous_node(message_service)  # Reuse (respects store_messages flag)
    prepare_timestamps = create_prepare_timestamps_node()  # Reuse as-is

    # Build the workflow graph
    workflow = StateGraph(LLMOnlyAnonymousState)

    # Add all nodes
    workflow.add_node("validate_inputs", validate_inputs)
    workflow.add_node("get_conversation_history_anonymous", get_conversation_history_anonymous)
    workflow.add_node("clean_message", clean_message)
    workflow.add_node("load_config", load_config)
    workflow.add_node("save_user_message_anonymous", save_user_message_anonymous)
    workflow.add_node("prepare_timestamps", prepare_timestamps)

    # Define the flow (much simpler than RAG)
    workflow.set_entry_point("validate_inputs")

    # Conditional edges
    workflow.add_conditional_edges(
        "validate_inputs",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {
            "continue": "get_conversation_history_anonymous",
            "stop": END
        }
    )

    workflow.add_edge("get_conversation_history_anonymous", "clean_message")
    workflow.add_edge("clean_message", "load_config")
    workflow.add_edge("load_config", "save_user_message_anonymous")

    workflow.add_conditional_edges(
        "save_user_message_anonymous",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {
            "continue": "prepare_timestamps",
            "stop": END
        }
    )

    workflow.add_edge("prepare_timestamps", END)

    return workflow


def initialize_llm_only_anonymous_workflow(
    session_factory: Any,
    message_service: Any,
    ia_config_service: Any
) -> None:
    """
    Initialize and compile the global LLM-only anonymous workflow.
    Should be called once at application startup.

    Args:
        session_factory: Database session factory (SessionLocal)
        message_service: Message service
        ia_config_service: IA config service
    """
    global _compiled_llm_only_anonymous_workflow

    logger.info("Compiling LLM-only anonymous workflow...")
    workflow = create_llm_only_anonymous_workflow(
        session_factory=session_factory,
        message_service=message_service,
        ia_config_service=ia_config_service
    )
    _compiled_llm_only_anonymous_workflow = workflow.compile()
    logger.info("LLM-only anonymous workflow compiled successfully")


def get_compiled_llm_only_anonymous_workflow() -> Any:
    """
    Get the compiled LLM-only anonymous workflow.
    Raises ValueError if workflow hasn't been initialized.

    Returns:
        Compiled StateGraph ready to execute
    """
    if _compiled_llm_only_anonymous_workflow is None:
        raise ValueError("LLM-only anonymous workflow not initialized. Call initialize_llm_only_anonymous_workflow() at startup.")
    return _compiled_llm_only_anonymous_workflow


# Re-export helpers for convenience
from app.workflows.helpers.service_helpers import (
    build_llm_only_initial_state_anonymous,
    validate_llm_only_workflow_state_anonymous
)
from app.workflows.helpers.nonstreaming_helpers import (
    execute_workflow_n8n,
    generate_complete_llm_only_response_anonymous
)

__all__ = [
    'LLMOnlyState',
    'create_llm_only_anonymous_workflow',
    'initialize_llm_only_anonymous_workflow',
    'get_compiled_llm_only_anonymous_workflow',
    'build_llm_only_initial_state_anonymous',
    'validate_llm_only_workflow_state_anonymous',
    'execute_workflow_n8n',
    'generate_complete_llm_only_response_anonymous',
]
