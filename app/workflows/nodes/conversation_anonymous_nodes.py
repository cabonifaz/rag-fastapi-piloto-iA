"""
Anonymous conversation-related nodes for workflows.
Handles anonymous conversation history.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_get_conversation_history_anonymous_node(message_service, max_ctx: int = 16):
    """Factory function to create get_conversation_history_anonymous node

    Args:
        message_service: Message service for DynamoDB
        max_ctx: Maximum context window (default 16 for RAG, use 8 for LLM-only)
    """
    async def get_conversation_history_anonymous(state) -> dict:
        """Fetch conversation history from DynamoDB for anonymous chat"""
        conversation_history = []
        chat_anonymous_id = state.get("chat_anonymous_id")

        if chat_anonymous_id is not None:
            try:
                messages = await message_service.get_last_n_messages_anonymous(
                    chat_anonymous_id=f"anon-{chat_anonymous_id}",
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
                logger.warning(f"Failed to retrieve anonymous chat history: {e}")

        state["conversation_history"] = conversation_history
        return state

    return get_conversation_history_anonymous
