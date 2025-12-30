"""
Port (interface) for RAG state builder services.
Defines the contract for building optimal query state using conversation history.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class StateBuilderPort(ABC):
    """
    Abstract interface for RAG state building.

    Implementations should build optimal query state by analyzing conversation history to:
    1. Resolve pronouns and implicit references
    2. Include necessary context from previous messages
    3. Create standalone, self-contained queries for better RAG retrieval
    """

    @abstractmethod
    async def build_query_state(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Build optimal query state for RAG using conversation history.

        Args:
            user_query: The user's query text
            conversation_history: Optional list of recent message dicts with 'role' and 'content'

        Returns:
            Dictionary with:
                - needs_context: bool (whether the query needed context)
                - response: str (the state-built query ready for RAG)
                - summary_intent: bool (whether user is asking for a summary)
        """
        pass
