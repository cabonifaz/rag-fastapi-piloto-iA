"""
Port (interface) for query rewriting services.
Defines the contract for rewriting and optimizing user queries for better RAG retrieval.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class QueryRewriterPort(ABC):
    """
    Abstract interface for query rewriting.

    Implementations should optimize user queries for RAG retrieval by:
    1. Using conversation state to add context
    2. Resolving ambiguous references
    3. Reformulating unclear queries
    4. Detecting summary requests
    """

    @abstractmethod
    async def rewrite_query(
        self,
        user_query: str,
        state: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Rewrite and optimize the user query using conversation state.

        Args:
            user_query: The user's query text to be rewritten
            state: Optional conversation state from state builder with {topic, entities, goal}

        Returns:
            Dictionary with:
                - needs_rewrite: bool (whether the query needed rewriting)
                - rewritten_query: str (the optimized query for RAG)
                - is_summary_request: bool (whether user is requesting a summary)
        """
        pass
