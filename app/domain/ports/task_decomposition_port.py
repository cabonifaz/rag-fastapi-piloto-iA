from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class TaskDecompositionPort(ABC):
    """
    Port interface for task decomposition services.
    Responsible for analyzing user queries and breaking them down into executable tasks.
    """

    @abstractmethod
    async def decompose_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        available_schemas: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decompose a user query into a list of executable tasks.

        Args:
            user_query: The user's question or request
            conversation_history: Previous conversation messages for context
            user_context: User information (company_id, area, etc.)
            available_schemas: Database table schemas if available

        Returns:
            List of task dictionaries with action and parameters

        Example:
            [
                {"action": "embedding", "input": "BGP protocol characteristics"},
                {"action": "retrieval", "vector_source": "embedding_result"},
                {"action": "llm_response", "instructions": ["explain clearly"]}
            ]
        """
        pass

    @abstractmethod
    async def analyze_intent(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user intent to determine query type and requirements.

        Args:
            user_query: The user's question or request
            conversation_history: Previous conversation messages for context

        Returns:
            Intent analysis with classification and metadata

        Example:
            {
                "intent_type": "vectorial_search",
                "requires_conversation": False,
                "requires_database": False,
                "requires_vectorial": True,
                "confidence": 0.95,
                "entities": ["BGP", "protocol"]
            }
        """
        pass

    @abstractmethod
    async def validate_task_chain(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that a task chain is properly formed and executable.

        Args:
            tasks: List of task dictionaries to validate

        Returns:
            Validation result with status and any errors

        Example:
            {
                "valid": True,
                "errors": [],
                "warnings": ["Task chain might take longer than usual"]
            }
        """
        pass