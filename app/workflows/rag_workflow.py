"""
LangGraph workflow for RAG query processing.
Modularizes the RAG pipeline into discrete, composable nodes.
"""
from typing import TypedDict, List, Dict, Any, Optional, AsyncGenerator
from langgraph.graph import StateGraph, END
import logging
import time
import asyncio
from datetime import datetime

from app.utils.query_utils import clean_user_query
from app.utils.search_utils import build_context_from_search_results
from app.utils.time_utils import format_timestamp_with_timezone
from app.infrastructure.llm.model_factory import ModelConfigFactory
from app.infrastructure.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)

# Global compiled workflow - initialized at startup
_compiled_rag_workflow: Optional[Any] = None


# Define State Schema
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


def create_rag_workflow(
    session_factory: Any,
    embeddings_provider: Any,
    vectorstore: Any,
    llm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    state_builder: Any,
    query_rewriter: Any
) -> StateGraph:
    """
    Create and configure the RAG workflow graph with dependencies captured in closures.

    Args:
        session_factory: Database session factory (SessionLocal) for getting sessions from pool
        embeddings_provider: Embeddings service
        vectorstore: Vector store service
        llm_provider: LLM service
        message_service: Message service
        ia_config_service: IA config service
        state_builder: State builder service
        query_rewriter: Query rewriter service

    Returns:
        Compiled StateGraph ready to execute
    """

    # Define all node functions as closures that capture dependencies
    async def validate_inputs(state: RAGState) -> RAGState:
        """Validate input parameters"""
        try:
            if not state["message"] or not state["message"].strip():
                raise ValueError("Message cannot be empty")
            if not state["user_id"]:
                raise ValueError("User ID is required")
            if not state["company_id"]:
                raise ValueError("Company ID is required")
            if not state["area_id"]:
                raise ValueError("Area is required")

            logger.info("Input validation successful")
            return state

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            state["error"] = str(e)
            state["should_stop"] = True
            return state

    async def get_conversation_history(state: RAGState) -> RAGState:
        """Fetch conversation history from DynamoDB"""
        conversation_history = []
        chat_id = state.get("chat_id")

        if chat_id is not None:
            try:
                messages = await message_service.get_last_n_messages(
                    chat_id=f"chat-{chat_id}",
                    n=20
                )

                if messages:
                    # Build and collapse in one pass
                    collapsed = []
                    for msg in messages:
                        role = "user" if msg.sender == 0 else "assistant"
                        if not collapsed or collapsed[-1]["role"] != role:
                            collapsed.append({"role": role, "content": msg.message})
                        else:
                            collapsed[-1]["content"] = msg.message

                    # Find boundaries
                    start = next((i for i, m in enumerate(collapsed) if m["role"] == "user"), -1)
                    end = next((i for i in range(len(collapsed)-1, -1, -1)
                               if collapsed[i]["role"] == "assistant"), -1)

                    # Validate and slice
                    if start >= 0 and end > start:
                        segment = collapsed[start:end+1]

                        # Quick alternating check
                        if all(segment[i]["role"] != segment[i+1]["role"]
                               for i in range(len(segment)-1)):

                            # Trim to context window (keep newest)
                            MAX_CTX = 16
                            if len(segment) > MAX_CTX:
                                trim = len(segment) - MAX_CTX
                                if segment[trim]["role"] == "assistant":
                                    trim -= 1
                                segment = segment[max(0, trim):]

                            # Final check
                            if segment and segment[0]["role"] == "user":
                                conversation_history = segment

            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {e}")

        state["conversation_history"] = conversation_history
        return state

    async def clean_message(state: RAGState) -> RAGState:
        """Clean user query"""
        state["cleaned_message"] = clean_user_query(state["message"])
        return state

    async def build_query_state(state: RAGState) -> RAGState:
        """Build query state for recontextualization"""
        chat_id = state.get("chat_id")
        conversation_history = state.get("conversation_history", [])

        state_builder_result = None

        if chat_id is not None and conversation_history:
            try:
                conversation_for_state_building = conversation_history[-6:] if len(conversation_history) >= 6 else conversation_history
                # Replace assistant messages content with placeholder
                conversation_for_state_building = [
                    {**msg, "content": "assistant message"} if msg["role"] == "assistant" else msg
                    for msg in conversation_for_state_building
                ]

                state_builder_result = await state_builder.build_query_state(
                    user_query=state["cleaned_message"],
                    conversation_history=conversation_for_state_building
                )
                logger.info(f"State builder result: {state_builder_result}")
            except Exception as e:
                logger.warning(f"Failed to build query state: {e}")

        state["state_builder_result"] = state_builder_result
        return state

    async def rewrite_query(state: RAGState) -> RAGState:
        """Rewrite query based on state"""
        query_rewriter_result = None
        state_builder_result = state.get("state_builder_result")

        if state_builder_result:
            try:
                query_rewriter_result = await query_rewriter.rewrite_query(
                    user_query=state["cleaned_message"],
                    state=state_builder_result
                )
                logger.info(f"Query rewriter result: {query_rewriter_result}")
            except Exception as e:
                logger.warning(f"Failed to rewrite query: {e}")

        state["query_rewriter_result"] = query_rewriter_result
        return state

    async def load_rag_config(state: RAGState) -> RAGState:
        """Load RAG configuration for the area"""
        # Get session from pool
        db = session_factory()
        try:
            rag_config = await ia_config_service.get_ia_area_config_rag(
                db,
                state["company_id"],
                state["area_id"]
            )
            logger.info(f"Retrieved RAG config: {rag_config}")
            state["rag_config"] = rag_config
            return state
        finally:
            db.close()  # Return session to pool

    async def create_or_use_chat(state: RAGState) -> RAGState:
        """Create a new chat if chat_id is not provided"""
        chat_id = state.get("chat_id")
        new_chat_created = False
        new_chat_titulo = None
        new_chat_timestamp = None

        if chat_id is None:
            now = datetime.now()
            formatted_date = now.strftime("%d/%m/%Y %H:%M")
            titulo = f"Nueva conversación {formatted_date}"

            # Get session from pool
            db = session_factory()
            try:
                chat_repository = ChatRepository(db)

                new_chat_id = await asyncio.to_thread(
                    chat_repository.create_chat,
                    id_usuario=state["user_id"],
                    id_area=state["area_id"],
                    id_empresa=state["company_id"],
                    titulo=titulo
                )

                if new_chat_id:
                    chat_id = new_chat_id
                    new_chat_created = True
                    new_chat_titulo = titulo
                    new_chat_timestamp = now.isoformat()
                else:
                    logger.error("Chat creation failed")
                    state["error"] = "Failed to create chat"
                    state["should_stop"] = True
                    return state
            finally:
                db.close()  # Return session to pool

        state["chat_id"] = chat_id
        state["new_chat_created"] = new_chat_created
        state["new_chat_titulo"] = new_chat_titulo
        state["new_chat_timestamp"] = new_chat_timestamp
        return state

    async def save_user_message(state: RAGState) -> RAGState:
        """Save user message to DynamoDB"""
        chat_id = state.get("chat_id")

        if chat_id:
            try:
                await message_service.create_message(
                    chat_id=chat_id,
                    created_at=state["created_at"],
                    sender=0,
                    message=state["cleaned_message"]
                )
            except Exception as e:
                logger.error(f"Failed to save user message: {e}")
                state["error"] = "Failed to save user message"
                state["should_stop"] = True

        return state

    async def determine_query_for_search(state: RAGState) -> RAGState:
        """Determine which query to use for embedding and search"""
        query_for_search = state["cleaned_message"]
        query_rewriter_result = state.get("query_rewriter_result")

        if query_rewriter_result and query_rewriter_result.get("needs_rewrite", False):
            rewritten_query = query_rewriter_result.get("rewritten_query", "").strip()
            if rewritten_query:
                query_for_search = rewritten_query
                logger.info(f"Using rewritten query for search: {query_for_search}")
            else:
                logger.warning("Rewritten query is empty, using original")
        else:
            logger.info("Using original query for search")

        state["query_for_search"] = query_for_search
        return state

    async def generate_embedding(state: RAGState) -> RAGState:
        """Generate embedding for the query"""
        query_embedding = await embeddings_provider.embed(state["query_for_search"])
        state["query_embedding"] = query_embedding
        return state

    async def search_vector_db(state: RAGState) -> RAGState:
        """Search vector database using hybrid search"""
        rag_config = state["rag_config"]

        search_results = await vectorstore.search_in_collection_hybrid(
            company_id=state["company_id"],
            area_id=state["area_id"],
            query_text=state["query_for_search"],
            query_vector=state["query_embedding"],
            top_k=rag_config['config']['RAG_TOP_K_RESULTS'],
            similarity_threshold=rag_config['config']['RAG_SIMILARITY_THRESHOLD'],
            alpha=rag_config['config']['RAG_ALPHA'],
            general_area=rag_config.get('general_area')
        )

        state["search_results"] = search_results
        return state

    async def build_context(state: RAGState) -> RAGState:
        """Build context text from search results"""
        context_text = build_context_from_search_results(state["search_results"])
        state["context_text"] = context_text
        return state

    async def select_history_for_prompt(state: RAGState) -> RAGState:
        """Select conversation history for LLM prompt based on flags"""
        conversation_history_for_prompt = []
        chat_id = state.get("chat_id")
        conversation_history = state.get("conversation_history", [])
        query_rewriter_result = state.get("query_rewriter_result")

        if chat_id and conversation_history and query_rewriter_result:
            needs_rewrite = query_rewriter_result.get("needs_rewrite", False)
            summary_intent = query_rewriter_result.get("is_summary_request", False)

            messages_to_use = 0
            if needs_rewrite and summary_intent:
                messages_to_use = 16
                logger.info("Using 16 messages for LLM prompt")
            elif needs_rewrite:
                messages_to_use = 8
                logger.info("Using 8 messages for LLM prompt")
            elif summary_intent:
                messages_to_use = 16
                logger.info("Using 16 messages for LLM prompt")

            if messages_to_use > 0:
                conversation_history_for_prompt = conversation_history[-messages_to_use:] if len(conversation_history) >= messages_to_use else conversation_history
                logger.info(f"Selected {len(conversation_history_for_prompt)} messages")

        state["conversation_history_for_prompt"] = conversation_history_for_prompt
        return state

    async def build_rag_prompt(state: RAGState) -> RAGState:
        """Build RAG prompt with context"""
        rag_config = state["rag_config"]
        model_config = ModelConfigFactory.get_model_config(rag_config['config']['LLM_MODEL'])
        rag_prompt = model_config.build_rag_prompt(state["cleaned_message"], state["context_text"])
        state["rag_prompt"] = rag_prompt
        return state

    async def prepare_timestamps(state: RAGState) -> RAGState:
        """Prepare timestamps for assistant message"""
        assistant_timestamp_ms = int(time.time() * 1000)
        user_timestamp_ms = int(state["created_at"])

        if assistant_timestamp_ms < user_timestamp_ms:
            assistant_timestamp = str(user_timestamp_ms + 1000)
            assistant_timestamp_ms = user_timestamp_ms + 1000
        else:
            assistant_timestamp = str(assistant_timestamp_ms)

        # Format timestamp with timezone
        utc_formatted, local_formatted = format_timestamp_with_timezone(
            assistant_timestamp_ms,
            state.get("request_timezone") or "America/Lima"
        )

        state["assistant_timestamp"] = assistant_timestamp
        state["assistant_timestamp_ms"] = assistant_timestamp_ms
        state["utc_formatted"] = utc_formatted
        state["local_formatted"] = local_formatted
        return state

    # Build the workflow graph
    workflow = StateGraph(RAGState)

    # Add all nodes
    workflow.add_node("validate_inputs", validate_inputs)
    workflow.add_node("get_conversation_history", get_conversation_history)
    workflow.add_node("clean_message", clean_message)
    workflow.add_node("build_query_state", build_query_state)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("load_rag_config", load_rag_config)
    workflow.add_node("create_or_use_chat", create_or_use_chat)
    workflow.add_node("save_user_message", save_user_message)
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
            "continue": "get_conversation_history",
            "stop": END
        }
    )

    workflow.add_edge("get_conversation_history", "clean_message")
    workflow.add_edge("clean_message", "build_query_state")
    workflow.add_edge("build_query_state", "rewrite_query")
    workflow.add_edge("rewrite_query", "load_rag_config")
    workflow.add_edge("load_rag_config", "create_or_use_chat")

    workflow.add_conditional_edges(
        "create_or_use_chat",
        lambda state: "stop" if state.get("should_stop", False) else "continue",
        {
            "continue": "save_user_message",
            "stop": END
        }
    )

    workflow.add_conditional_edges(
        "save_user_message",
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


def initialize_rag_workflow(
    session_factory: Any,
    embeddings_provider: Any,
    vectorstore: Any,
    llm_provider: Any,
    message_service: Any,
    ia_config_service: Any,
    state_builder: Any,
    query_rewriter: Any
) -> None:
    """
    Initialize and compile the global RAG workflow.
    Should be called once at application startup.

    Args:
        session_factory: Database session factory (SessionLocal)
        embeddings_provider: Embeddings service
        vectorstore: Vector store service
        llm_provider: LLM service
        message_service: Message service
        ia_config_service: IA config service
        state_builder: State builder service
        query_rewriter: Query rewriter service
    """
    global _compiled_rag_workflow

    logger.info("Compiling RAG workflow...")
    workflow = create_rag_workflow(
        session_factory=session_factory,
        embeddings_provider=embeddings_provider,
        vectorstore=vectorstore,
        llm_provider=llm_provider,
        message_service=message_service,
        ia_config_service=ia_config_service,
        state_builder=state_builder,
        query_rewriter=query_rewriter
    )
    _compiled_rag_workflow = workflow.compile()
    logger.info("RAG workflow compiled successfully")


def get_compiled_rag_workflow() -> Any:
    """
    Get the compiled RAG workflow.
    Raises ValueError if workflow hasn't been initialized.

    Returns:
        Compiled StateGraph ready to execute
    """
    if _compiled_rag_workflow is None:
        raise ValueError("RAG workflow not initialized. Call initialize_rag_workflow() at startup.")
    return _compiled_rag_workflow


# ============================================================================
# Helper Functions for Service Layer
# ============================================================================

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


async def stream_workflow_progress(app: Any, initial_state: RAGState):
    """
    Execute workflow and yield progress updates.
    Generator that yields progress events during execution, then yields final state.
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


async def stream_llm_response(state: RAGState, llm_provider: Any, message_service: Any, db: Any):
    """
    Stream LLM response and handle message saving.
    Consolidates all streaming logic including validation, error handling, and persistence.
    """
    from app.infrastructure.repositories.chat_repository import ChatRepository

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

    # Validate parameters
    if messages is None and (not prompt or not prompt.strip()):
        raise ValueError("Either prompt or messages must be provided")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

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
                chat_repo = ChatRepository(db)
                await asyncio.to_thread(
                    chat_repo.update_ultimo_mensaje_fecha,
                    chat_id
                )
                first_chunk_sent = True

            yield {
                "type": "chunk",
                "content": chunk
            }

        # Validate that content was generated
        if not has_content:
            raise ValueError("El modelo no generó una respuesta. Por favor, intenta reformular tu pregunta.")

        # Save assistant message to DynamoDB
        if chat_id and assistant_response and state["assistant_timestamp"]:
            await message_service.create_message(
                chat_id=chat_id,
                created_at=state["assistant_timestamp"],
                sender=1,
                message=assistant_response
            )

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
