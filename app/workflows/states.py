"""
State definitions for LangGraph workflows.
Centralized state schemas used across different workflows.
"""
from typing import TypedDict, List, Dict, Any, Optional


class RAGState(TypedDict):
    """State schema for RAG workflow - only request data, no dependencies"""
    # Input parameters
    user_id: int
    message: str
    company_id: int
    area_id: int
    created_at: str
    chat_id: Optional[str]
    request_timezone: Optional[str]

    # Processing state
    cleaned_message: Optional[str]
    conversation_history: List[Dict[str, str]]
    state_builder_result: Optional[Dict[str, Any]]
    query_rewriter_result: Optional[Dict[str, Any]]
    rag_config: Optional[Dict[str, Any]]
    new_chat_created: bool
    new_chat_titulo: Optional[str]
    new_chat_timestamp: Optional[str]
    query_for_search: Optional[str]
    query_embedding: Optional[List[float]]
    search_results: Optional[List[Dict[str, Any]]]
    context_text: Optional[str]
    conversation_history_for_prompt: List[Dict[str, str]]
    rag_prompt: Optional[str]
    assistant_timestamp: Optional[str]
    assistant_timestamp_ms: Optional[int]
    utc_formatted: Optional[str]
    local_formatted: Optional[str]

    # Error handling
    error: Optional[str]
    should_stop: bool


class LLMOnlyState(TypedDict):
    """State schema for LLM-only workflow (no RAG) - only request data, no dependencies"""
    # Input parameters
    user_id: int
    message: str
    company_id: int
    created_at: str
    chat_id: str
    system_behavior: Optional[str]
    custom_llm: Optional[str]
    request_timezone: Optional[str]
    use_guidelines: bool
    store_messages: bool

    # Processing state
    cleaned_message: Optional[str]
    conversation_history: List[Dict[str, str]]
    llm_config: Optional[Dict[str, Any]]
    assistant_timestamp: Optional[str]
    assistant_timestamp_ms: Optional[int]
    utc_formatted: Optional[str]
    local_formatted: Optional[str]

    # Error handling
    error: Optional[str]
    should_stop: bool
