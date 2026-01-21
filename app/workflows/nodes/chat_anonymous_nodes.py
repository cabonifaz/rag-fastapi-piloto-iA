"""
Anonymous chat management nodes for workflows.
Handles anonymous chat message saving.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_save_user_message_anonymous_node(message_service):
    """Factory function to create save_user_message_anonymous node"""
    async def save_user_message_anonymous(state) -> dict:
        """Save user message to DynamoDB for anonymous chat (respects store_messages flag if present)"""
        chat_anonymous_id = state.get("chat_anonymous_id")
        store_messages = state.get("store_messages", True)  # Default True for RAG workflows

        if chat_anonymous_id and store_messages:
            try:
                await message_service.create_message_anonymous(
                    chat_anonymous_id=chat_anonymous_id,
                    created_at=state["created_at"],
                    sender=0,
                    message=state["cleaned_message"]
                )
            except Exception as e:
                logger.error(f"Failed to save anonymous user message: {e}")
                state["error"] = "Failed to save anonymous user message"
                state["should_stop"] = True
        elif not store_messages:
            logger.info("Skipping anonymous user message storage (store_messages=False)")

        return state

    return save_user_message_anonymous


def create_save_original_message_anonymous_node(message_service):
    """Factory function to create save_original_message_anonymous node.
    Saves the original message (not cleaned_message which contains the processed rag_query).
    Used in RAG anonymous workflow where message is stored and rag_query is processed.
    """
    async def save_original_message_anonymous(state) -> dict:
        """Save original user message to DynamoDB for anonymous chat"""
        chat_anonymous_id = state.get("chat_anonymous_id")

        if chat_anonymous_id:
            try:
                await message_service.create_message_anonymous(
                    chat_anonymous_id=chat_anonymous_id,
                    created_at=state["created_at"],
                    sender=0,
                    message=state["message"]
                )
            except Exception as e:
                logger.error(f"Failed to save original anonymous user message: {e}")
                state["error"] = "Failed to save original anonymous user message"
                state["should_stop"] = True

        return state

    return save_original_message_anonymous
