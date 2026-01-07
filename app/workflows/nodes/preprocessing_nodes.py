"""
Preprocessing nodes for workflows.
Handles message cleaning and configuration loading.
"""
import logging
from app.workflows.states import RAGState
from app.utils.query_utils import clean_user_query

logger = logging.getLogger(__name__)


def create_clean_message_node():
    """Factory function to create clean_message node"""
    async def clean_message(state: RAGState) -> RAGState:
        """Clean user query"""
        state["cleaned_message"] = clean_user_query(state["message"])
        return state

    return clean_message


def create_load_rag_config_node(session_factory, ia_config_service):
    """Factory function to create load_rag_config node"""
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

    return load_rag_config
