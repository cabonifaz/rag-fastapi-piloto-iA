"""
Conversation-related nodes for workflows.
Handles conversation history, state building, and query rewriting.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_get_conversation_history_node(message_service, max_ctx: int = 16):
    """Factory function to create get_conversation_history node

    Args:
        message_service: Message service for DynamoDB
        max_ctx: Maximum context window (default 16 for RAG, use 8 for LLM-only)
    """
    async def get_conversation_history(state) -> dict:
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
                            if len(segment) > max_ctx:
                                trim = len(segment) - max_ctx
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

    return get_conversation_history


def create_build_query_state_node(state_builder):
    """Factory function to create build_query_state node"""
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

    return build_query_state


def create_rewrite_query_node(query_rewriter):
    """Factory function to create rewrite_query node"""
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

    return rewrite_query
