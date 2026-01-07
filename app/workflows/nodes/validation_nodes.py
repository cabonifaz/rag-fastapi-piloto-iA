"""
Validation nodes for workflows.
Handles input validation and error detection.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_validate_inputs_node():
    """Factory function to create validate_inputs node"""
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

    return validate_inputs
