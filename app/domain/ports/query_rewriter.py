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
    1. Expanding abbreviations and technical terms
    2. Adding relevant domain-specific keywords
    3. Reformulating complex questions into clearer search queries
    4. Generating multiple query variations for better coverage
    """

    @abstractmethod
    async def rewrite_query(
        self,
        user_query: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Rewrite and optimize the user query for better RAG retrieval.

        Args:
            user_query: The user's query text to be rewritten
            domain_context: Optional domain/industry context for better query optimization

        Returns:
            Dictionary with:
                - rewritten_query: str (the optimized query for RAG)
                - variations: List[str] (alternative query formulations)
                - keywords: List[str] (extracted/added keywords)
        """
        pass
