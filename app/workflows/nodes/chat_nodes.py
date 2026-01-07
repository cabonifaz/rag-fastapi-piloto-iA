"""
Chat management nodes for workflows.
Handles chat creation and message saving.
"""
import logging
import asyncio
from datetime import datetime
from app.workflows.states import RAGState
from app.infrastructure.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)


def create_create_or_use_chat_node(session_factory):
    """Factory function to create create_or_use_chat node"""
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

    return create_or_use_chat


def create_save_user_message_node(message_service):
    """Factory function to create save_user_message node"""
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

    return save_user_message
