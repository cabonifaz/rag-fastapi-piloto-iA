"""
Port (interface) for query recontextualization services.
Defines the contract for recontextualizing user queries using conversation history.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class RecontextualizerPort(ABC):
    """
    Abstract interface for query recontextualization.

    Implementations should recontextualize user queries by analyzing conversation history to:
    1. Resolve pronouns and implicit references
    2. Include necessary context from previous messages
    3. Create standalone, self-contained queries for better RAG retrieval
    """

    @abstractmethod
    async def recontextualize_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Recontextualize the user query using conversation history.

        Args:
            user_query: The user's query text
            conversation_history: Optional list of recent message dicts with 'role' and 'content'

        Returns:
            Dictionary with:
                - needs_context: bool (whether the query needed context)
                - response: str (the recontextualized query)
                - summary_intent: bool (whether user is asking for a summary)
        """
        pass
