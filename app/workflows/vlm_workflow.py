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
    create_get_conversation_history_node,
    create_clean_message_node,
    create_context_gatekeeper_node,
    create_recontextualize_query_node,
    create_load_rag_config_node,
    create_create_or_use_chat_node,
    create_save_user_message_node,
    create_select_history_for_prompt_node,
    create_prepare_timestamps_node,
    create_build_vlm_prompt_node,
)

logger = logging.getLogger(__name__)

# Global compiled workflow - initialized at startup
_compiled_vlm_workflow: Optional[Any] = None


def create_vlm_workflow(
    session_factory: Any,
    vlm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    context_gatekeeper: Any,
    recontextualizer: Any,
) -> StateGraph:
    """
    Create and configure the VLM workflow graph with dependencies.

    Args:
        session_factory: Database session factory (SessionLocal)
        vlm_provider: VLM provider (AWSBedrockVLMProvider)
        message_service: Message service
        ia_config_service: IA config service
        context_gatekeeper: Context gatekeeper service
        recontextualizer: Query recontextualizer service

    Returns:
        Compiled StateGraph ready to execute
    """
    validate_inputs            = create_validate_inputs_node()
    get_conversation_history   = create_get_conversation_history_node(message_service)
    clean_message              = create_clean_message_node()
    context_gatekeeper_node    = create_context_gatekeeper_node(context_gatekeeper)
    recontextualize_query      = create_recontextualize_query_node(recontextualizer)
    load_rag_config            = create_load_rag_config_node(session_factory, ia_config_service)
    create_or_use_chat         = create_create_or_use_chat_node(session_factory)
    save_user_message          = create_save_user_message_node(message_service)
    select_history_for_prompt  = create_select_history_for_prompt_node()
    build_vlm_prompt           = create_build_vlm_prompt_node()
    prepare_timestamps         = create_prepare_timestamps_node()

    workflow = StateGraph(VLMState)

    workflow.add_node("validate_inputs",           validate_inputs)
    workflow.add_node("get_conversation_history",  get_conversation_history)
    workflow.add_node("clean_message",             clean_message)
    workflow.add_node("context_gatekeeper",        context_gatekeeper_node)
    workflow.add_node("recontextualize_query",     recontextualize_query)
    workflow.add_node("load_rag_config",           load_rag_config)
    workflow.add_node("create_or_use_chat",        create_or_use_chat)
    workflow.add_node("save_user_message",         save_user_message)
    workflow.add_node("select_history_for_prompt", select_history_for_prompt)
    workflow.add_node("build_vlm_prompt",          build_vlm_prompt)
    workflow.add_node("prepare_timestamps",        prepare_timestamps)

    workflow.set_entry_point("validate_inputs")

    workflow.add_conditional_edges(
        "validate_inputs",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "get_conversation_history", "stop": END}
    )

    workflow.add_edge("get_conversation_history", "clean_message")
    workflow.add_edge("clean_message", "context_gatekeeper")

    workflow.add_conditional_edges(
        "context_gatekeeper",
        lambda state: "recontextualize" if state.get("gatekeeper_result", {}).get("needs_context") else "skip",
        {"recontextualize": "recontextualize_query", "skip": "load_rag_config"}
    )

    workflow.add_edge("recontextualize_query", "load_rag_config")
    workflow.add_edge("load_rag_config", "create_or_use_chat")

    workflow.add_conditional_edges(
        "create_or_use_chat",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "save_user_message", "stop": END}
    )

    workflow.add_conditional_edges(
        "save_user_message",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {"continue": "select_history_for_prompt", "stop": END}
    )

    # No embedding/search/context nodes — goes straight to prompt building
    workflow.add_edge("select_history_for_prompt", "build_vlm_prompt")
    workflow.add_edge("build_vlm_prompt",          "prepare_timestamps")
    workflow.add_edge("prepare_timestamps",        END)

    return workflow


def initialize_vlm_workflow(
    session_factory: Any,
    vlm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    context_gatekeeper: Any,
    recontextualizer: Any,
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
        ia_config_service=ia_config_service,
        context_gatekeeper=context_gatekeeper,
        recontextualizer=recontextualizer,
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
