"""
Validation nodes for workflows.
Handles input validation and error detection.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_validate_inputs_node(require_area_id: bool = True):
    """Factory function to create validate_inputs node

    Args:
        require_area_id: Whether area_id is required (False for LLM-only workflows)
    """
    async def validate_inputs(state) -> dict:
        """Validate input parameters"""
        try:
            if not state["message"] or not state["message"].strip():
                raise ValueError("Message cannot be empty")
            if not state["user_id"]:
                raise ValueError("User ID is required")
            if not state["company_id"]:
                raise ValueError("Company ID is required")
            if require_area_id and not state.get("area_id"):
                raise ValueError("Area is required")

            logger.info("Input validation successful")
            return state

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            state["error"] = str(e)
            state["should_stop"] = True
            return state

    return validate_inputs
