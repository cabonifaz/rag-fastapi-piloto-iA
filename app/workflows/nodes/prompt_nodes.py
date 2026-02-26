"""
Prompt building nodes for workflows.
Handles conversation history selection, RAG prompt building, and timestamp preparation.
"""
import logging
import time
from app.workflows.states import RAGState
from app.infrastructure.llm.model_factory import ModelConfigFactory
from app.utils.time_utils import format_timestamp_with_timezone

logger = logging.getLogger(__name__)


def create_select_history_for_prompt_node():
    """Factory function to create select_history_for_prompt node"""
    async def select_history_for_prompt(state: RAGState) -> RAGState:
        """Select conversation history for LLM prompt based on flags"""
        conversation_history_for_prompt = []
        chat_id = state.get("chat_id")
        conversation_history = state.get("conversation_history", [])
        gatekeeper_result = state.get("gatekeeper_result", {})

        if chat_id and conversation_history and gatekeeper_result:
            needs_context = gatekeeper_result.get("needs_context", False)
            is_summary = gatekeeper_result.get("is_summary", False)

            messages_to_use = 0
            if is_summary:
                messages_to_use = 16
                logger.info("Using 16 messages for LLM prompt")
            elif needs_context:
                messages_to_use = 8
                logger.info("Using 8 messages for LLM prompt")
            else:
                messages_to_use = 0
                logger.info("Using 0 messages for LLM prompt")

            if messages_to_use > 0:
                conversation_history_for_prompt = conversation_history[-messages_to_use:] if len(conversation_history) >= messages_to_use else conversation_history
                logger.info(f"Selected {len(conversation_history_for_prompt)} messages")

        state["conversation_history_for_prompt"] = conversation_history_for_prompt
        return state

    return select_history_for_prompt


def create_build_rag_prompt_node():
    """Factory function to create build_rag_prompt node"""
    async def build_rag_prompt(state: RAGState) -> RAGState:
        """Build RAG prompt with context"""
        rag_config = state["rag_config"]
        model_config = ModelConfigFactory.get_model_config(rag_config['config']['LLM_MODEL'])
        rag_prompt = model_config.build_rag_prompt(state["cleaned_message"], state["context_text"])
        state["rag_prompt"] = rag_prompt
        return state

    return build_rag_prompt


def create_prepare_timestamps_node():
    """Factory function to create prepare_timestamps node"""
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

    return prepare_timestamps
