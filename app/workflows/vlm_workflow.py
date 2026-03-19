"""
LangGraph workflow for VLM query processing.
Assembles modular nodes into a complete VLM streaming workflow.
Replaces the vector search branch with a multimodal prompt build step.
"""
from typing import Any, Optional
from langgraph.graph import StateGraph, END
import logging

from app.workflows.states import VLMState
from app.workflows.nodes import (
    create_validate_inputs_node,
    create_clean_message_node,
    create_create_or_use_chat_node,
    create_build_attachment_keys_node,
    create_save_vlm_user_message_node,
    create_build_vlm_prompt_node,
    create_prepare_timestamps_node,
)

logger = logging.getLogger(__name__)

# Global compiled workflow - initialized at startup
_compiled_vlm_workflow: Optional[Any] = None


def create_vlm_workflow(
    session_factory: Any,
    vlm_provider: Any,
    message_service: Any,
) -> StateGraph:
    """
    Create and configure the VLM workflow graph with dependencies.

    Args:
        session_factory: Database session factory (SessionLocal)
        vlm_provider: VLM provider (AWSBedrockVLMProvider)
        message_service: Message service

    Returns:
        Compiled StateGraph ready to execute
    """
    validate_inputs       = create_validate_inputs_node()
    clean_message         = create_clean_message_node()
    create_or_use_chat    = create_create_or_use_chat_node(session_factory)
    build_attachment_keys = create_build_attachment_keys_node()
    save_user_message     = create_save_vlm_user_message_node(message_service)
    build_vlm_prompt      = create_build_vlm_prompt_node()
    prepare_timestamps    = create_prepare_timestamps_node()

    workflow = StateGraph(VLMState)

    workflow.add_node("validate_inputs",       validate_inputs)
    workflow.add_node("clean_message",         clean_message)
    workflow.add_node("create_or_use_chat",    create_or_use_chat)
    workflow.add_node("build_attachment_keys", build_attachment_keys)
    workflow.add_node("save_user_message",     save_user_message)
    workflow.add_node("build_vlm_prompt",      build_vlm_prompt)
    workflow.add_node("prepare_timestamps",    prepare_timestamps)

    workflow.set_entry_point("validate_inputs")

    workflow.add_conditional_edges(
        "validate_inputs",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "clean_message", "stop": END}
    )

    workflow.add_edge("clean_message",         "create_or_use_chat")

    workflow.add_conditional_edges(
        "create_or_use_chat",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "build_attachment_keys", "stop": END}
    )

    workflow.add_edge("build_attachment_keys", "save_user_message")

    workflow.add_conditional_edges(
        "save_user_message",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "build_vlm_prompt", "stop": END}
    )

    workflow.add_edge("build_vlm_prompt",   "prepare_timestamps")
    workflow.add_edge("prepare_timestamps", END)

    return workflow


def initialize_vlm_workflow(
    session_factory: Any,
    vlm_provider: Any,
    message_service: Any,
) -> None:
    """
    Initialize and compile the global VLM workflow.
    Should be called once at application startup.
    """
    global _compiled_vlm_workflow

    logger.info("Compiling VLM workflow...")
    workflow = create_vlm_workflow(
        session_factory=session_factory,
        vlm_provider=vlm_provider,
        message_service=message_service,
    )
    _compiled_vlm_workflow = workflow.compile()
    logger.info("VLM workflow compiled successfully")


def get_compiled_vlm_workflow() -> Any:
    """
    Get the compiled VLM workflow.
    Raises ValueError if workflow hasn't been initialized.
    """
    if _compiled_vlm_workflow is None:
        raise ValueError("VLM workflow not initialized. Call initialize_vlm_workflow() at startup.")
    return _compiled_vlm_workflow
