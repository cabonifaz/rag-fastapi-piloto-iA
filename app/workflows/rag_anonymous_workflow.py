"""
LangGraph workflow for RAG query processing with anonymous chats.
Assembles modular nodes into a complete RAG streaming workflow for anonymous users.
"""
from typing import Any, Optional
from langgraph.graph import StateGraph, END
import logging

from app.workflows.states import RAGAnonymousState
from app.workflows.nodes import (
    create_recontextualize_query_node,
    create_compare_query_node,
    create_load_rag_config_node,
    create_determine_query_for_search_node,
    create_generate_embedding_node,
    create_search_vector_db_node,
    create_build_context_node,
    create_select_history_for_prompt_node,
    create_build_rag_prompt_node,
    create_prepare_timestamps_node
)
from app.workflows.nodes.validation_nodes import create_validate_inputs_node_rag_anonymous
from app.workflows.nodes.conversation_anonymous_nodes import create_get_conversation_history_anonymous_node
from app.workflows.nodes.chat_anonymous_nodes import create_save_original_message_anonymous_node
from app.workflows.nodes.preprocessing_nodes import create_clean_rag_query_node

logger = logging.getLogger(__name__)

# Global compiled workflow - initialized at startup
_compiled_rag_anonymous_workflow: Optional[Any] = None


def create_rag_anonymous_workflow(
    session_factory: Any,
    embeddings_provider: Any,
    vectorstore: Any,
    llm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    recontextualizer: Any,
    comparator: Any
) -> StateGraph:
    """
    Create and configure the RAG anonymous workflow graph with dependencies.

    Args:
        session_factory: Database session factory (SessionLocal) for getting sessions from pool
        embeddings_provider: Embeddings service
        vectorstore: Vector store service
        llm_provider: LLM service
        message_service: Message service
        ia_config_service: IA config service
        recontextualizer: Query recontextualizer service
        comparator: Query comparator service

    Returns:
        Compiled StateGraph ready to execute
    """
    # Create nodes using factory functions
    validate_inputs = create_validate_inputs_node_rag_anonymous()
    get_conversation_history_anonymous = create_get_conversation_history_anonymous_node(message_service)
    clean_rag_query = create_clean_rag_query_node()
    recontextualize_query = create_recontextualize_query_node(recontextualizer)
    compare_query = create_compare_query_node(comparator)
    load_rag_config = create_load_rag_config_node(session_factory, ia_config_service)
    save_original_message_anonymous = create_save_original_message_anonymous_node(message_service)
    determine_query_for_search = create_determine_query_for_search_node()
    generate_embedding = create_generate_embedding_node(embeddings_provider)
    search_vector_db = create_search_vector_db_node(vectorstore)
    build_context = create_build_context_node()
    select_history_for_prompt = create_select_history_for_prompt_node()
    build_rag_prompt = create_build_rag_prompt_node()
    prepare_timestamps = create_prepare_timestamps_node()

    # Build the workflow graph
    workflow = StateGraph(RAGAnonymousState)

    # Add all nodes
    workflow.add_node("validate_inputs", validate_inputs)
    workflow.add_node("get_conversation_history_anonymous", get_conversation_history_anonymous)
    workflow.add_node("clean_rag_query", clean_rag_query)
    workflow.add_node("recontextualize_query", recontextualize_query)
    workflow.add_node("compare_query", compare_query)
    workflow.add_node("load_rag_config", load_rag_config)
    workflow.add_node("save_original_message_anonymous", save_original_message_anonymous)
    workflow.add_node("determine_query_for_search", determine_query_for_search)
    workflow.add_node("generate_embedding", generate_embedding)
    workflow.add_node("search_vector_db", search_vector_db)
    workflow.add_node("build_context", build_context)
    workflow.add_node("select_history_for_prompt", select_history_for_prompt)
    workflow.add_node("build_rag_prompt", build_rag_prompt)
    workflow.add_node("prepare_timestamps", prepare_timestamps)

    # Define the flow
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

    workflow.add_edge("get_conversation_history_anonymous", "clean_rag_query")
    workflow.add_edge("clean_rag_query", "recontextualize_query")
    workflow.add_edge("recontextualize_query", "compare_query")
    workflow.add_edge("compare_query", "load_rag_config")
    workflow.add_edge("load_rag_config", "save_original_message_anonymous")

    workflow.add_conditional_edges(
        "save_original_message_anonymous",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {
            "continue": "determine_query_for_search",
            "stop": END
        }
    )

    workflow.add_edge("determine_query_for_search", "generate_embedding")
    workflow.add_edge("generate_embedding", "search_vector_db")
    workflow.add_edge("search_vector_db", "build_context")
    workflow.add_edge("build_context", "select_history_for_prompt")
    workflow.add_edge("select_history_for_prompt", "build_rag_prompt")
    workflow.add_edge("build_rag_prompt", "prepare_timestamps")
    workflow.add_edge("prepare_timestamps", END)

    return workflow


def initialize_rag_anonymous_workflow(
    session_factory: Any,
    embeddings_provider: Any,
    vectorstore: Any,
    llm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    recontextualizer: Any,
    comparator: Any
) -> None:
    """
    Initialize and compile the global RAG anonymous workflow.
    Should be called once at application startup.

    Args:
        session_factory: Database session factory (SessionLocal)
        embeddings_provider: Embeddings service
        vectorstore: Vector store service
        llm_provider: LLM service
        message_service: Message service
        ia_config_service: IA config service
        recontextualizer: Query recontextualizer service
        comparator: Query comparator service
    """
    global _compiled_rag_anonymous_workflow

    logger.info("Compiling RAG anonymous workflow...")
    workflow = create_rag_anonymous_workflow(
        session_factory=session_factory,
        embeddings_provider=embeddings_provider,
        vectorstore=vectorstore,
        llm_provider=llm_provider,
        message_service=message_service,
        ia_config_service=ia_config_service,
        recontextualizer=recontextualizer,
        comparator=comparator
    )
    _compiled_rag_anonymous_workflow = workflow.compile()
    logger.info("RAG anonymous workflow compiled successfully")


def get_compiled_rag_anonymous_workflow() -> Any:
    """
    Get the compiled RAG anonymous workflow.
    Raises ValueError if workflow hasn't been initialized.

    Returns:
        Compiled StateGraph ready to execute
    """
    if _compiled_rag_anonymous_workflow is None:
        raise ValueError("RAG anonymous workflow not initialized. Call initialize_rag_anonymous_workflow() at startup.")
    return _compiled_rag_anonymous_workflow


# ============================================================================
# Re-export helpers for convenience
# ============================================================================
from app.workflows.helpers.service_helpers import (
    build_initial_state_anonymous,
    validate_workflow_state_anonymous
)
from app.workflows.helpers.nonstreaming_helpers import (
    generate_complete_llm_response_anonymous,
    execute_workflow_n8n
)

__all__ = [
    'create_rag_anonymous_workflow',
    'initialize_rag_anonymous_workflow',
    'get_compiled_rag_anonymous_workflow',
    'build_initial_state_anonymous',
    'validate_workflow_state_anonymous',
    'generate_complete_llm_response_anonymous',
    'execute_workflow_n8n',
]
